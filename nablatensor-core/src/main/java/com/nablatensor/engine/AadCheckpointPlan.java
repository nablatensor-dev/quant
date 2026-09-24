/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.engine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;

/**
 * Segment-checkpoint analysis and layout shared by device adjoint code
 * generators.
 *
 * <p>A fully unrolled reverse sweep keeps every forward value {@code v_i} live
 * until the reverse consumes it — for a 1500-node tape that is a live set no GPU
 * register file holds, so the shader spills to scratch memory and occupancy
 * collapses. This plan cuts the tape into {@code segments} contiguous pieces of
 * ~{@code sqrt(nodes)} nodes each. The forward runs piece by piece and, at each
 * internal boundary, saves only the handful of values that later nodes still
 * read (the <em>carry set</em>). The reverse then walks the pieces backwards:
 * for each it restores the boundary carry, recomputes that piece's forward into
 * a small local set, and runs the piece's reverse. Peak live set drops from
 * {@code O(nodes)} to {@code O(sqrt(nodes))} at the cost of recomputing the
 * forward once.
 *
 * <p>The carry set at boundary {@code b} is {@code C[b] = { i < b : some node
 * j >= b reads i as an argument }}. A carry node whose last use is at or beyond
 * the final internal boundary is <em>global</em>: it stays in a register for
 * the whole kernel body and is never written to the checkpoint buffer.
 */
public final class AadCheckpointPlan {

  public final int nodes;
  /** {@code bound[0] == 0}, {@code bound[segments] == nodes}, strictly increasing. */
  public final int[] bound;
  public final int segments;
  /** {@code lastUse[i]} = greatest node index that reads {@code i}; {@code nodes} for the output; {@code -1} if unread. */
  public final int[] lastUse;
  /** {@code carry[s]} = sorted {@code C[bound[s]]} for {@code s} in {@code 1..segments-1}; {@code carry[0]} empty. */
  public final int[][] carry;
  /** Nodes kept in a register for the whole kernel (carried across every internal boundary). */
  public final boolean[] global;
  /** Float offset into a path's checkpoint row where {@code carry[s] \ global} starts. */
  public final int[] sliceOffset;
  /** {@code carry[s] \ global}, i.e. the nodes segment {@code s} actually writes to / reads from the checkpoint buffer. */
  public final int[][] slice;
  /** Total floats per path in the checkpoint buffer. */
  public final int slotsPerPath;

  /** Generic segment boundaries and carry sets for backends with their own storage strategy. */
  public record Analysis(int[] bound, int[] lastUse, int[][] carry) {
    public int segments() {
      return bound.length - 1;
    }

    /** Nodes that cross at least one boundary. */
    public boolean[] carryUnionMask() {
      boolean[] union = new boolean[lastUse.length];
      for (int s = 1; s < carry.length; s++) {
        for (int node : carry[s]) union[node] = true;
      }
      return union;
    }
  }

  private AadCheckpointPlan(int nodes, int[] bound, int[] lastUse, int[][] carry,
                             boolean[] global, int[] sliceOffset, int[][] slice, int slotsPerPath) {
    this.nodes = nodes;
    this.bound = bound;
    this.segments = bound.length - 1;
    this.lastUse = lastUse;
    this.carry = carry;
    this.global = global;
    this.sliceOffset = sliceOffset;
    this.slice = slice;
    this.slotsPerPath = slotsPerPath;
  }

  public static boolean nodeArgA(AadOpEnum op) {
    return switch (op) {
      case ADD, SUB, MUL, DIV, NEG, EXP, LOG, SQRT, ABS, MAX, MIN -> true;
      case CONST, INPUT, RANDN, RANDU -> false;
    };
  }

  public static boolean nodeArgB(AadOpEnum op) {
    return switch (op) {
      case ADD, SUB, MUL, DIV, MAX, MIN -> true;
      default -> false;
    };
  }

  /** Computes legal cuts near equal-sized targets, preferring cuts with fewer crossing values. */
  public static Analysis balanced(AadTape tape, int requestedSegments) {
    int n = tape.size();
    int segments = Math.max(1, Math.min(requestedSegments, n));
    int[] provisional = new int[segments + 1];
    for (int s = 0; s <= segments; s++) {
      provisional[s] = (int) ((long) s * n / segments);
    }
    int[] lastUse = lastUses(tape);
    int[] bounds = new int[segments + 1];
    bounds[segments] = n;
    for (int s = 1; s < segments; s++) {
      int target = provisional[s];
      int window = Math.max(1, n / (4 * segments));
      int low = Math.max(bounds[s - 1] + 1, target - window);
      int high = Math.min(n - 1, target + window);
      int best = Math.max(low, Math.min(high, target));
      int bestSize = Integer.MAX_VALUE;
      for (int c = low; c <= high; c++) {
        int size = 0;
        for (int j = 0; j < c; j++) {
          if (lastUse[j] >= c) size++;
        }
        if (size < bestSize || (size == bestSize && Math.abs(c - target) < Math.abs(best - target))) {
          best = c;
          bestSize = size;
        }
      }
      bounds[s] = best;
    }
    return analyze(tape, bounds, lastUse);
  }

  /** Computes last uses and the values crossing each supplied segment boundary. */
  public static Analysis analyze(AadTape tape, int[] bounds) {
    return analyze(tape, bounds, lastUses(tape));
  }

  private static Analysis analyze(AadTape tape, int[] bounds, int[] lastUse) {
    if (bounds.length < 2 || bounds[0] != 0 || bounds[bounds.length - 1] != tape.size()) {
      throw new IllegalArgumentException("checkpoint bounds must start at 0 and end at tape size");
    }
    for (int s = 1; s < bounds.length; s++) {
      if (bounds[s] <= bounds[s - 1]) {
        throw new IllegalArgumentException("checkpoint bounds must be strictly increasing");
      }
    }
    int[][] carry = new int[bounds.length - 1][];
    carry[0] = new int[0];
    for (int s = 1; s < bounds.length - 1; s++) {
      int b = bounds[s];
      List<Integer> nodes = new ArrayList<>();
      for (int i = 0; i < b; i++) {
        if (lastUse[i] >= b) nodes.add(i);
      }
      carry[s] = nodes.stream().mapToInt(Integer::intValue).toArray();
    }
    return new Analysis(bounds.clone(), lastUse, carry);
  }

  private static int[] lastUses(AadTape tape) {
    int n = tape.size();
    int[] lastUse = new int[n];
    Arrays.fill(lastUse, -1);
    for (int j = 0; j < n; j++) {
      AadOpEnum op = tape.op(j);
      if (nodeArgA(op)) {
        lastUse[tape.argA(j)] = j;
      }
      if (nodeArgB(op)) {
        lastUse[tape.argB(j)] = j;
      }
    }
    lastUse[tape.outputNode()] = n;
    return lastUse;
  }

  /**
   * Builds a plan, or returns {@code null} when checkpointing does not apply:
   * value-only kernels, tapes at or below {@code minNodes}, tapes that would
   * split into fewer than two segments, or tapes whose output node is not the
   * final node / is not differentiable (the unrolled path handles those, and
   * they are never the large tapes checkpointing exists for).
   */
  public static AadCheckpointPlan of(AadTape tape, AadOptions options, int minNodes) {
    if (!options.adjoints()) {
      return null;
    }
    int n = tape.size();
    if (n <= Math.max(2, minNodes)) {
      return null;
    }
    if (!tape.isActive(tape.outputNode())) {
      return null;
    }
    int segLen = Integer.getInteger("nablatensor.checkpoint.segLen",
        Math.max(48, (int) Math.round(Math.sqrt(n))));
    segLen = Math.max(4, segLen);
    int segCount = (n + segLen - 1) / segLen;
    if (segCount < 2) {
      return null;
    }
    int[] bound = new int[segCount + 1];
    for (int s = 1; s < segCount; s++) {
      bound[s] = Math.min(n, s * segLen);
    }
    bound[segCount] = n;
    // guard against a degenerate final boundary collision
    for (int s = 1; s <= segCount; s++) {
      if (bound[s] <= bound[s - 1]) {
        return null;
      }
    }

    int[] lastUse = lastUses(tape);

    if (tape.outputNode() < bound[segCount - 1]) {
      return null;
    }

    int lastInternal = bound[segCount - 1];
    boolean[] global = new boolean[n];
    for (int i = 0; i < n; i++) {
      global[i] = lastUse[i] >= lastInternal && i < lastInternal;
    }

    Analysis sharedAnalysis = analyze(tape, bound, lastUse);
    int[][] carry = sharedAnalysis.carry();
    int[][] slice = new int[segCount][];
    int[] sliceOffset = new int[segCount];
    int running = 0;
    for (int s = 1; s < segCount; s++) {
      List<Integer> sl = new ArrayList<>();
      for (int i : carry[s]) {
        if (!global[i]) sl.add(i);
      }
      slice[s] = sl.stream().mapToInt(Integer::intValue).toArray();
      sliceOffset[s] = running;
      running += slice[s].length;
    }

    return new AadCheckpointPlan(n, bound, sharedAnalysis.lastUse(), sharedAnalysis.carry(),
        global, sliceOffset, slice, running);
  }

  /** Union of every {@code carry[s]}: the nodes that need a persistent reverse accumulator. */
  public int[] carryUnion() {
    boolean[] seen = new boolean[nodes];
    int count = 0;
    for (int s = 1; s < segments; s++) {
      for (int i : carry[s]) {
        if (!seen[i]) {
          seen[i] = true;
          count++;
        }
      }
    }
    int[] out = new int[count];
    int k = 0;
    for (int i = 0; i < nodes; i++) {
      if (seen[i]) {
        out[k++] = i;
      }
    }
    return out;
  }

  /** The checkpoint slot for node {@code i} at boundary {@code s}, or {@code -1} if it is global / not in the slice. */
  public int slotOf(int s, int node) {
    int[] sl = slice[s];
    for (int k = 0; k < sl.length; k++) {
      if (sl[k] == node) {
        return sliceOffset[s] + k;
      }
    }
    return -1;
  }

  /** Value reference for node {@code i}: a whole-kernel register for a global carry node, a segment local otherwise. */
  public String vref(int i) {
    return (global[i] ? "g_v" : "v") + i;
  }

  /** Segment index containing node {@code i}. */
  public int segmentOf(int i) {
    int s = 0;
    while (bound[s + 1] <= i) {
      s++;
    }
    return s;
  }

  private boolean[] carryMembership;

  public boolean inCarry(int i) {
    if (carryMembership == null) {
      carryMembership = new boolean[nodes];
      for (int s = 1; s < segments; s++) {
        for (int node : carry[s]) {
          carryMembership[node] = true;
        }
      }
    }
    return carryMembership[i];
  }

  public enum CudaOpEmissionEnum {
    CONST(AadOpEnum.CONST, "{CONST}", null, null, null, null),
    INPUT(AadOpEnum.INPUT, "{INPUT}", null, null, null, null),
    RANDN(AadOpEnum.RANDN, "{RANDN}", null, null, null, null),
    RANDU(AadOpEnum.RANDU, "{RANDU}", null, null, null, null),
    ADD(AadOpEnum.ADD, "{A} + {B}", "{DI}", "{DI}", null, null),
    SUB(AadOpEnum.SUB, "{A} - {B}", "{DI}", "-({DI})", null, null),
    MUL(AadOpEnum.MUL, "{A} * {B}", "{DI} * {B}", "{DI} * {A}", null, null),
    DIV(AadOpEnum.DIV, "{A} / {B}", "{DI} / {B}", "-({DI} * {VI} / {B})", null, null),
    NEG(AadOpEnum.NEG, "-{A}", "-({DI})", null, null, null),
    EXP(AadOpEnum.EXP, "{EXP}({A})", "{DI} * {VI}", null, null, null),
    LOG(AadOpEnum.LOG, "{LOG}({A})", "{DI} / {A}", null, null, null),
    SQRT(AadOpEnum.SQRT, "{SQRT}({A})", "{DI} * {HALF} / {VI}", null, null, null),
    ABS(AadOpEnum.ABS, "{ABS}({A})", "({A} < {ZERO} ? -{DI} : {DI})", null, null, null),
    MAX(AadOpEnum.MAX, "{FMAX}({A}, {B})", "{DI}", "{DI}", "{A} >= {B}", "!({A} >= {B})"),
    MIN(AadOpEnum.MIN, "{FMIN}({A}, {B})", "{DI}", "{DI}", "{A} <= {B}", "!({A} <= {B})");

    private static final EnumMap<AadOpEnum, CudaOpEmissionEnum> BY_OP = new EnumMap<>(AadOpEnum.class);
    static { for (CudaOpEmissionEnum rule : values()) BY_OP.put(rule.op, rule); }
    private final AadOpEnum op;
    private final String forwardTemplate, gradientA, gradientB, conditionA, conditionB;
    CudaOpEmissionEnum(AadOpEnum op, String forwardTemplate, String gradientA, String gradientB,
                       String conditionA, String conditionB) {
      this.op=op; this.forwardTemplate=forwardTemplate; this.gradientA=gradientA; this.gradientB=gradientB;
      this.conditionA=conditionA; this.conditionB=conditionB;
    }
    public static CudaOpEmissionEnum forOp(AadOpEnum op) { return BY_OP.get(op); }
    public String forward(String constant, String input, String randn, String randu, String a, String b, boolean f32) {
      String exp=f32?"__expf":"exp", log=f32?"logf":"log", sqrt=f32?"sqrtf":"sqrt", abs=f32?"fabsf":"fabs";
      return render(forwardTemplate, constant,input,randn,randu,a==null?"0":a,b==null?"0":b,"0","0","0",f32,exp,log,sqrt,abs);
    }
    public void appendCheckpointReverse(StringBuilder src, AadTape tape, int i, boolean f32,
                                 AadCheckpointPlan plan, int segLo) {
      int a=tape.argA(i), b=tape.argB(i);
      boolean activeA=a>=0&&tape.isActive(a), activeB=b>=0&&tape.isActive(b);
      String da=a>=0?(a>=segLo?"d"+a:"dc_"+a):null, db=b>=0?(b>=segLo?"d"+b:"dc_"+b):null;
      String va=a>=0?plan.vref(a):null, vb=b>=0?plan.vref(b):null;
      appendGradient(src,activeA,da,gradientA,conditionA,va,vb,"d"+i,plan.vref(i),f32);
      appendGradient(src,activeB,db,gradientB,conditionB,va,vb,"d"+i,plan.vref(i),f32);
    }
    private static void appendGradient(StringBuilder src, boolean active, String target, String formula,
                                      String condition, String a, String b, String di, String vi, boolean f32) {
      if (!active || formula==null) return;
      String zero=f32?"0.0f":"0.0";
      String rendered=render(formula,"0","0","0","0",a==null?"0":a,b==null?"0":b,di,vi,zero,f32,"0","0","0","0");
      String predicate=condition==null?null:render(condition,"0","0","0","0",a==null?"0":a,b==null?"0":b,di,vi,zero,f32,"0","0","0","0");
      src.append("      ");
      if(predicate!=null) src.append("if (").append(predicate).append(") ");
      src.append(target).append(" += ").append(rendered).append(";\n");
    }
    private static String render(String template,String constant,String input,String randn,String randu,
                                 String a,String b,String di,String vi,String zero,boolean f32,
                                 String exp,String log,String sqrt,String abs) {
      return template.replace("{CONST}",constant).replace("{INPUT}",input).replace("{RANDN}",randn).replace("{RANDU}",randu)
          .replace("{A}",a).replace("{B}",b).replace("{DI}",di).replace("{VI}",vi).replace("{ZERO}",zero)
          .replace("{HALF}",f32?"0.5f":"0.5").replace("{EXP}",exp).replace("{LOG}",log).replace("{SQRT}",sqrt)
          .replace("{ABS}",abs).replace("{FMAX}",f32?"fmaxf":"fmax").replace("{FMIN}",f32?"fminf":"fmin");
    }
  }
}

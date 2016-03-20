package com.spbsu.ml.models.nn.nfa;

import com.spbsu.commons.math.vectors.Mx;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.VecTools;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import com.spbsu.ml.FuncC1;
import com.spbsu.ml.func.generic.Const;
import com.spbsu.ml.func.generic.SubVecFuncC1;
import com.spbsu.ml.func.generic.WSum;
import com.spbsu.ml.models.nn.NeuralSpider;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.spbsu.commons.math.vectors.VecTools.scale;
import static com.spbsu.commons.math.vectors.VecTools.subtract;

/**
 * Created by afonin.s on 06.11.2015.
 */
public class NFANetworkWithLZW<T> extends NeuralSpider<T, Seq<T>> {
  private final FastRandom rng;
  private final double dropout;
  final int statesCount;
  private final Seq<T> alpha;
  private final int dim;
  private final int transitionMxDim;
  private final int finalStates;
  private final ThreadLocal<Map<Seq<T>, SeqWeightsCalculator>> dictionary;


  public NFANetworkWithLZW(FastRandom rng, double dropout, int statesCount, int finalStates, Seq<T> alpha) {
    super();
    this.rng = rng;
    this.dropout = dropout;
    this.statesCount = statesCount;
    this.finalStates = finalStates;
    this.alpha = alpha;
    transitionMxDim = (statesCount - finalStates) * (statesCount - 1);
    dim = transitionMxDim * alpha.length();
    dictionary = new ThreadLocal<Map<Seq<T>, SeqWeightsCalculator>>() {
      @Override
      protected Map<Seq<T>, SeqWeightsCalculator> initialValue() {
        final Map<Seq<T>, SeqWeightsCalculator> result = new HashMap<>(alpha.length());
        for (int i = 0; i < alpha.length(); i++) {
          result.put(alpha.sub(i, i + 1), new SeqWeightsCalculator(statesCount, finalStates, transitionMxDim, i * transitionMxDim));
        }
        return result;
      }
    };
  }

  @Override
  protected Topology topology(Seq<T> seq, boolean dropout) {
    final int capacity = (seq.length() + 1) * statesCount + finalStates + 1;
    final List<Node> nodes = new ArrayList<>(capacity);
    for (int i = 0; i < statesCount; i++) {
      final Const aConst = new Const(i == 0 ? 1 : 0);
      nodes.add(new InputNode(aConst));
    }
    final boolean[] dropoutArr = new boolean[statesCount];

    if (dropout && rng.nextDouble() < this.dropout)
      dropoutArr[rng.nextInt(statesCount - finalStates - 1) + 1] = true;

    int start = 0, end = 0;
    for (int d = 0; d < seq.length() + 1; d++) {
      if (end != seq.length())
        end++;
      Seq<T> sub = seq.sub(start, end);

      SeqWeightsCalculator calcer = dictionary.get().get(sub);
      if (calcer == null || sub.length() > 3) {
        final int subLength = sub.length();
        final int[] wStarts = new int[subLength];
        for (int i = 0; i < subLength; i++) {
          wStarts[i] = SeqTools.indexOf(alpha, sub.at(i)) * transitionMxDim;
        }
        dictionary.get().put(sub, new SeqWeightsCalculator(statesCount, finalStates, transitionMxDim, wStarts));

        final Seq<T> prevSub = sub.sub(0, subLength - 1);
        final SeqWeightsCalculator prevCalcer = dictionary.get().get(prevSub);
        //System.out.println("put into network LAYER: " + prevSub);
        fillLayer(nodes, prevSub, prevCalcer, dropoutArr, capacity);

        start += prevSub.length();
      } else if (end == seq.length()) {
        fillLayer(nodes, sub, calcer, dropoutArr, capacity);
      }
    }
    final int lastLayerStart = nodes.size() - statesCount;
    nodes.add(new NonDeterminedNode(statesCount - finalStates, lastLayerStart, capacity));
    for (int i = 0; i < finalStates; i++) {
      nodes.add(new OutputNode(nodes.toArray(new Node[] {}), statesCount, statesCount - finalStates + i));
    }

    return new NFATopology<>(statesCount, finalStates, dropout, nodes.toArray(new Node[] {}), dropoutArr);
  }

  private void fillLayer(List<Node> nodes, Seq<T> sub,
                         SeqWeightsCalculator calcer, boolean[] dropoutArr, int capacity) {
    final int prevLayerStart = nodes.size() - statesCount;
    //System.out.println("LAYER " + sub + " start from pStart: " + prevLayerStart);
    //final int prevLayerStart = d * statesCount;
    calcer.setDropOut(dropoutArr);
    for (int i = 0; i < statesCount; i++) {
      nodes.add(new MyNode(i, dim, prevLayerStart, statesCount, capacity, calcer));
    }
  }

  @Override
  public int dim() {
    return dim;
  }

  public String ppState(Vec state, Seq<T> seq) {
    final StringBuilder builder = new StringBuilder();
    builder.append(dictionary.get().keySet()).append('\n');

    List<Seq<T>> subs = new ArrayList<>(seq.length());
    int start = 0, end = 0;
    for (int d = 0; d <= seq.length(); d++) {
      if (end != seq.length())
        end++;

//      System.out.println(start + " - " + end);
      Seq<T> sub = seq.sub(start, end);
      SeqWeightsCalculator calcer = dictionary.get().get(sub);
      if (calcer == null) {
        subs.add(seq.sub(start, end - 2));
        start = end - 2;
      } else if (end == seq.length()) {
        subs.add(sub.sub(0, sub.length() - 1));
        subs.add(sub.sub(sub.length() - 1, sub.length()));
        break;
      }
    }

    builder.append(subs).append('\n');
    for (int i = 0; i < state.length(); i++) {
      if (i % statesCount == 0) {
        builder.append('\n');
        int i1 = i / statesCount;
        if (i1 > 0 && i1 <= subs.size()) {
          builder.append(subs.get(i1 - 1));
        }
      }

      builder.append("\t").append(CharSeqTools.prettyPrint.format(state.get(i)));
    }
//    builder.append(" ");
//    for (int i = (subs.size() + 1) * statesCount; i < state.length(); i++) {
//      builder.append("\t").append(CharSeqTools.prettyPrint.format(state.get(i)));
//    }
    builder.append('\n');
    return builder.toString();
  }

  protected class MyNode implements NeuralSpider.Node {
    private final int index;
    private final int wDim;
    private final int pStart;
    private final int pLen;
    private final int maxCapacity;
    private final SeqWeightsCalculator calcer;

    public MyNode(int index, int wDim, int pStart, int pLen, int maxCapacity, SeqWeightsCalculator calcer) {
      this.index = index;
      this.wDim = wDim;
      this.pStart = pStart;
      this.pLen = pLen;
      this.maxCapacity = maxCapacity;
      this.calcer = calcer;
    }

    @Override
    public FuncC1 transByParents(Vec parents) {
      return new FuncC1.Stub() {

        @Override
        public Vec gradientTo(Vec betta, Vec to) {
          return calcer.gradientTo(betta, to, parents.sub(pStart, pLen), index);
        }

        @Override
        public double value(Vec betta) {
          final Mx weights = calcer.compute(betta);
          return VecTools.multiply(weights.row(index), parents.sub(pStart, pLen));
        }

        @Override
        public int dim() {
          return wDim;
        }
      };
    }

    public FuncC1 transByParameters(Vec betta) {
      final Mx weights = calcer.compute(betta);
      return new SubVecFuncC1(new WSum(weights.row(index)), pStart, pLen, maxCapacity);
    }
  }
}

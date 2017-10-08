package com.spbsu.ml.models.nn;

import com.spbsu.commons.math.FuncC1;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.ml.func.generic.Const;

import java.util.*;

import static com.spbsu.ml.models.nn.NetBuilder.DATA_LAYER_NAME;

public class ConvNet {
  private final NeuralSpider.Node[] nodes;
  private final LinkedHashMap<String, NetBuilder.Layer> layerHashMap;
  private final boolean[] isDroppable;
  private final int[] layersNodeSize;
  private final int outputCount;
  private final Random rng;
  private final float dropout;
  private final NeuralSpider<Double, Vec> spider;
  private final Vec weights;
  private final int numLayers;

  public ConvNet(ArrayList<Integer> layersNodeSize,
                 LinkedHashMap<String, NetBuilder.Layer> layerHashMap,
                 Random rng, float dropout, List<NeuralSpider.Node> nodes,
                 int weightSize) {
    this.layerHashMap = layerHashMap;
    Iterator<NetBuilder.Layer> it = layerHashMap.values().iterator();
    numLayers = layerHashMap.size();
    isDroppable = new boolean[numLayers];
    this.layersNodeSize = new int[numLayers];
    this.rng = rng;
    this.dropout = dropout;
    this.nodes = new NeuralSpider.Node[nodes.size()];
    nodes.toArray(this.nodes);

    weights = new ArrayVec(new double[weightSize]);

    spider = new NeuralSpider<Double, Vec>() {
      @Override
      protected NeuralSpider.Topology topology(Vec argument, boolean isDrop) {
        final NetBuilder.Layer dataLayer = layerHashMap.get(DATA_LAYER_NAME);
        final int numInputs = dataLayer.getView().stateLength;
        if (argument.dim() != numInputs)
          throw new IllegalArgumentException();

        final NeuralSpider.Node[] inputLayer = new NeuralSpider.Node[numInputs];
        for(int i = 0; i < inputLayer.length; i++) {
          final int nindex = i;
          inputLayer[i] = new NeuralSpider.Node() {
            @Override
            public FuncC1 transByParameters(Vec betta) {
              return new Const(argument.get(nindex));
            }

            @Override
            public FuncC1 transByParents(Vec state) {
              return new Const(argument.get(nindex));
            }
          };
        }

        return new NeuralSpider.Topology.Stub() {
          @Override
          public int outputCount() {
            return outputCount;
          }

          @Override
          public boolean isDroppedOut(int nodeIndex) {
            //noinspection SimplifiableIfStatement
            if (!isDrop || !isDroppable(nodeIndex))
              return false;
            return dropout > MathTools.EPSILON && rng.nextDouble() < dropout;
          }

          @Override
          public NeuralSpider.Node at(int i) {
            return i <= inputLayer.length ? inputLayer[i - 1] : ConvNet.this.nodes[i - inputLayer.length - 1];
          }

          @Override
          public int length() {
            return inputLayer.length + ConvNet.this.nodes.length + 1;
          }
        };
      }

      @Override
      public int dim() {
        return weightSize;
      }
    };

    for (int i = 0; i < numLayers; i++) {
      final NetBuilder.Layer layer = it.next();
      isDroppable[i] = !(layer instanceof NetBuilder.Layer3D) && !layer.getName().equals(DATA_LAYER_NAME);
      this.layersNodeSize[i] = layersNodeSize.get(i);
    }

    outputCount = this.layersNodeSize[this.layersNodeSize.length - 1];
  }

  private int getLayerIndex(int nodeIndex) {
    for (int i = 0; i < layersNodeSize.length; i++) {
      if (layersNodeSize[i] > nodeIndex)
        return i;
      nodeIndex -= layersNodeSize[i];
    }

    throw new IndexOutOfBoundsException();
  }

  private boolean isDroppable(int nodeIndex) {
    return isDroppable[getLayerIndex(nodeIndex)];
  }

  public Vec forward(Vec input) {
    return spider.compute(input, weights);
  }

  public void initialize(WeightInitializer initializer) {
    for (Iterator<NetBuilder.Layer> iterator = layerHashMap.values().iterator();
         iterator.hasNext();) {
      final NetBuilder.Layer layer = iterator.next();
      Vec subWeights = weights.sub(layer.getView().weightStart, layer.getView().weightLength);
      initializer.apply(subWeights);
    }
  }
}

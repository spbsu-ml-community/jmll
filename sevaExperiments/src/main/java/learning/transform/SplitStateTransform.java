package learning.transform;

import com.spbsu.commons.seq.regexp.Alphabet;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.set.hash.TIntHashSet;
import learning.LearningState;

public class SplitStateTransform<T> implements Transform<T> {
  private final int state;
  private final int alphabetSize;

  public SplitStateTransform(int state, int alphabetSize) {
    this.state = state;
    this.alphabetSize = alphabetSize;
  }

  @Override
  public LearningState<T> applyTransform(LearningState<T> learningState) {
    LearningState<T> result = new LearningState<>(learningState);
    final Alphabet<T> alphabet = learningState.getAlphabet();
    for (int c = 0; c < alphabetSize; c++) {
      if (!result.getAutomaton().hasTransition(state, alphabet.getT(alphabet.get(c)))) {
        result.getStateClassWeight().add(new double[learningState.getClassCount()]);
        result.getSamplesEndState().add(new TIntIntHashMap());
        result.getSamplesViaState().add(new TIntHashSet());
        int newState = result.getAutomaton().createNewState();
        result = new AddTransitionTransform<T>(state, newState, alphabet.getT(alphabet.get(c))).applyTransform(result);
      }
    }
    return result;
  }

  @Override
  public String getDescription() {
    return "Split state " + state;
  }
}

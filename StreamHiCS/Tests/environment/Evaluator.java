package environment;

import subspace.Subspace;
import subspace.SubspaceSet;

public class Evaluator {

	public static double evaluateTPvsFP(SubspaceSet result, SubspaceSet correctResult) {
		System.out.println("Expected: " + correctResult.toString());
		int l = correctResult.size();
		int tp = 0;
		for (Subspace s : correctResult.getSubspaces()) {
			if (result.contains(s)) {
				tp++;
			}
		}
		int fp = result.size() - tp;
		double recall = 1;
		double fpRatio = 0;
		if (l > 0) {
			recall = ((double) tp) / correctResult.size();
			fpRatio = ((double) fp) / correctResult.size();
		} else {
			if (fp > 0) {
				recall = 0;
				fpRatio = 1;
			}
		}
		
		System.out.println("Correlated: " + result.toString());
		if (!result.isEmpty()) {
			for (Subspace s : result.getSubspaces()) {
				System.out.print(s.getContrast() + ", ");
			}
			System.out.println();
		}
		System.out.println("True positives: " + tp + " out of " + correctResult.size() + "; False positives: " + fp);
		double score = recall - fpRatio;
		System.out.println("Score: " + score);
		
		System.out.println();
		return score;
	}
}

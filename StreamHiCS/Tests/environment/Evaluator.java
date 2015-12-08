package environment;

import subspace.Subspace;
import subspace.SubspaceSet;

public class Evaluator {

	public static void displayResult(SubspaceSet result, SubspaceSet correctResult){
		System.out.println("Expected: " + correctResult.toString());
		System.out.println("Correlated: " + result.toString());
		if (!result.isEmpty()) {
			for (Subspace s : result.getSubspaces()) {
				System.out.print(s.getContrast() + ", ");
			}
			System.out.println();
		}
	}
	
	public static double evaluateTPvsFP(SubspaceSet result, SubspaceSet correctResult) {
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
		
		System.out.println("True positives: " + tp + " out of " + correctResult.size() + "; False positives: " + fp);
		double score = recall - fpRatio;
		System.out.println("Score: " + score);
		
		System.out.println();
		return score;
	}
	
	/**
	 * Calculates the average Jaccard-index. 
	 * 
	 * @param result
	 * @param correctResult
	 * @return
	 */
	public static double evaluateJaccardIndex(SubspaceSet result, SubspaceSet correctResult){
		double jaccardIndex = 0;
		double maxJaccardIndex = 0;
		double sumMaxJaccardIndex = 0;
		for(Subspace sr : result.getSubspaces()){
			jaccardIndex = 0;
			for(Subspace scr : correctResult.getSubspaces()){
				jaccardIndex = jaccardIndex(sr, scr);
				if(jaccardIndex > maxJaccardIndex){
					maxJaccardIndex = jaccardIndex;
				}
			}
			sumMaxJaccardIndex += maxJaccardIndex;
		}
		
		double averageMaxJaccardIndex = sumMaxJaccardIndex / result.size();
		
		System.out.println("Average max Jaccard-Index: " + averageMaxJaccardIndex);
		
		return averageMaxJaccardIndex;
	}
	
	private static double jaccardIndex(Subspace s1, Subspace s2){
		double jaccardIndex = 0;
		if(s1.isEmpty() && s2.isEmpty()){
			jaccardIndex = 1;
		}else if(s1.isEmpty() || s2.isEmpty()){
			jaccardIndex = 0;
		}else{
			int cut = s1.cut(s2);
			jaccardIndex = ((double) cut) / (s1.size() + s2.size() - cut);
		}
		return jaccardIndex;
	}
}

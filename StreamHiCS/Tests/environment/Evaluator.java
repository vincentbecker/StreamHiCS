package environment;

import java.util.ArrayList;

import subspace.Subspace;
import subspace.SubspaceSet;

public class Evaluator {

	public static void displayResult(SubspaceSet result, SubspaceSet correctResult) {
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

		// System.out.println("True positives: " + tp + " out of " +
		// correctResult.size() + "; False positives: " + fp);
		double score = recall - fpRatio;
		// System.out.println("TPvsFP-score: " + score);
		return score;
	}

	/**
	 * Calculates the average Jaccard-index.
	 * 
	 * @param result
	 * @param correctResult
	 * @return
	 */
	public static double evaluateJaccardIndex(SubspaceSet result, SubspaceSet correctResult) {
		double jaccardIndex = 0;
		double maxJaccardIndex = 0;
		double sumMaxJaccardIndex = 0;
		for (Subspace sr : result.getSubspaces()) {
			maxJaccardIndex = 0;
			for (Subspace scr : correctResult.getSubspaces()) {
				jaccardIndex = jaccardIndex(sr, scr);
				if (jaccardIndex > maxJaccardIndex) {
					maxJaccardIndex = jaccardIndex;
				}
			}
			sumMaxJaccardIndex += maxJaccardIndex;
		}

		double averageMaxJaccardIndex = 0;
		if (result.isEmpty()) {
			if (correctResult.isEmpty()) {
				averageMaxJaccardIndex = 1;
			} else {
				averageMaxJaccardIndex = 0;
			}
		} else {
			averageMaxJaccardIndex = sumMaxJaccardIndex / result.size();
		}

		// System.out.println("Average max Jaccard-index: " +
		// averageMaxJaccardIndex);

		return averageMaxJaccardIndex;
	}

	/**
	 * Calculates the average Jaccard-index.
	 * 
	 * @param result
	 * @param correctResult
	 * @return
	 */
	public static double evaluateStructuralSimilarity(SubspaceSet result, SubspaceSet correctResult) {
		double ss = 0;
		double maxSS = 0;
		double sumMaxSS = 0;
		for (Subspace sr : result.getSubspaces()) {
			maxSS = 0;
			for (Subspace scr : correctResult.getSubspaces()) {
				ss = structuralSimilarity(sr, scr);
				if (ss > maxSS) {
					maxSS = ss;
				}
			}
			sumMaxSS += maxSS;
		}

		double averageMaxSS = 0;
		if (result.isEmpty()) {
			if (correctResult.isEmpty()) {
				averageMaxSS = 1;
			} else {
				averageMaxSS = 0;
			}
		} else {
			averageMaxSS = sumMaxSS / result.size();
		}

		// System.out.println("Average max structural similarity: " +
		// averageMaxSS);

		return averageMaxSS;
	}

	private static double jaccardIndex(Subspace s1, Subspace s2) {
		double jaccardIndex = 0;
		if (s1.isEmpty() && s2.isEmpty()) {
			jaccardIndex = 1;
		} else if (s1.isEmpty() || s2.isEmpty()) {
			jaccardIndex = 0;
		} else {
			int cut = s1.cut(s2);
			jaccardIndex = ((double) cut) / (s1.size() + s2.size() - cut);
		}
		return jaccardIndex;
	}

	private static double structuralSimilarity(Subspace s1, Subspace s2) {
		double ss = 0;
		if (s1.isEmpty() && s2.isEmpty()) {
			ss = 1;
		} else if (s1.isEmpty() || s2.isEmpty()) {
			ss = 0;
		} else {
			int cut = s1.cut(s2);
			ss = ((double) cut) / (Math.sqrt(s1.size() * s2.size()));
		}
		return ss;
	}

	public static double[] evaluateConceptChange(ArrayList<Double> trueChanges, ArrayList<Double> detectedChanges,
			int changeLength, int streamLength) {
		double sumTimeToDetection = 0;
		int missedDetections = 0;
		int t = trueChanges.size();
		int d = detectedChanges.size();

		// ArrayList<Double> falseAlarms = new ArrayList<Double>();
		int falseAlarms = 0;
		double detectedChange = 0;
		int i = 0;
		int j = 0;
		boolean[] trueChangesFound = new boolean[t];
		for (i = 0; i < d; i++) {
			detectedChange = detectedChanges.get(i);
			while (j < t && trueChanges.get(j) < detectedChange) {
				j++;
			}
			if (j == 0) {
				// falseAlarms.add(detectedChange);
				falseAlarms++;
			} else if (!trueChangesFound[j - 1]) {
				trueChangesFound[j - 1] = true;
				sumTimeToDetection += (detectedChange - trueChanges.get(j - 1));
			} else if (detectedChange - trueChanges.get(j - 1) < changeLength) {
				// We don't treat these as false positives
				// Do not do anything
			} else {
				falseAlarms++;
				// falseAlarms.add(detectedChange);
			}
		}

		for (j = 0; j < t; j++) {
			if (!trueChangesFound[j]) {
				missedDetections++;
			}
		}

		double mtd = 0;
		double mdr = 1;
		if (t > 0) {
			if (t - missedDetections > 0) {
				mtd = sumTimeToDetection / (t - missedDetections);
			}
			mdr = ((double) missedDetections) / t;
		}
		double mtfa = ((double) streamLength) / (falseAlarms + 1);

		/*
		 * int f = falseAlarms.size(); double sumTimeBetweenFalseAlarms = 0;
		 * double previousFalseAlarm = 0; double falseAlarm = 0; i = 0; for (i =
		 * 0; i < f; i++) { falseAlarm = falseAlarms.get(i);
		 * sumTimeBetweenFalseAlarms += (falseAlarm - previousFalseAlarm);
		 * previousFalseAlarm = falseAlarm; } sumTimeBetweenFalseAlarms +=
		 * (streamLength - falseAlarm); mtfa = sumTimeBetweenFalseAlarms / f;
		 */

		double mtr = 0;
		if (mtd != 0) {
			mtr = mtfa / mtd * (1 - mdr);
		}

		double[] results = new double[4];
		results[0] = mtfa;
		results[1] = mtd;
		results[2] = mdr;
		results[3] = mtr;

		return results;
	}
}

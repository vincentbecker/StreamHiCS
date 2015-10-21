package centroids;

import java.util.ArrayList;

/**
 * Implements a kind of Page-Hinkley test for the {@link Centroid}s. However,
 * the incoming instances this test is carried out on are not single
 * observations, but the whole centroid set, we receive every time the check
 * method is called. Another difference is that we have a multivariate stream,
 * and not a single so we will use the norm of the difference of mT and UT as
 * test statistic.
 * 
 * @author Vincent
 *
 */
public class PHTChecker extends ChangeChecker {

	private int numberOfDimensions;
	private double magnitudeThreshold;
	private double detectionThreshold;
	private double increasingMT = Double.MAX_VALUE;
	private double decreasingMT = Double.MIN_VALUE;
	// private double checkCounter = 0;
	private double[] overallWeightedSum;
	private double overallTotalWeight = 0;

	public PHTChecker(int numberOfDimensions, double magnitudeThreshold, double detectionThreshold) {
		this.numberOfDimensions = numberOfDimensions;
		this.magnitudeThreshold = magnitudeThreshold;
		this.detectionThreshold = detectionThreshold;
		overallWeightedSum = new double[numberOfDimensions];
	}

	@Override
	public boolean checkForChange(ArrayList<Centroid> centroids) {
		double[] currentWeightedSum = new double[numberOfDimensions];
		double[] currentWeightedMean = new double[numberOfDimensions];
		double totalWeight = 0;
		double[] vector;
		double weight;
		for (Centroid c : centroids) {
			vector = c.getVector();
			weight = c.getWeight();
			totalWeight += weight;
			for (int i = 0; i < numberOfDimensions; i++) {
				currentWeightedSum[i] += vector[i] * weight;
			}
		}

		// Calculate the mean of the current centroids
		for (int i = 0; i < numberOfDimensions; i++) {
			currentWeightedMean[i] = currentWeightedSum[i] / totalWeight;
		}

		// Add the currentWeightedSum to weightedSum and the totalWeigth to
		// sumTotalWeight
		for (int i = 0; i < numberOfDimensions; i++) {
			overallWeightedSum[i] += currentWeightedSum[i];
		}
		overallTotalWeight += totalWeight;

		// Calculate mt by calculating the deviation from the overall mean and
		// of that the euclidian norm
		double[] deviation = new double[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			deviation[i] = currentWeightedMean[i] - overallWeightedSum[i] / overallTotalWeight;
		}
		double mt = euclidianNorm(deviation) - magnitudeThreshold;

		increasingMT = Math.min(increasingMT, mt);
		decreasingMT = Math.max(decreasingMT, mt);

		double increasingPHT = mt - increasingMT;
		double decreasingPHT = decreasingMT - mt;
		System.out.println("IncreasingPHT: " + increasingPHT + ", DecreasingPHT: " + decreasingPHT);
		if (increasingPHT > detectionThreshold || decreasingPHT > detectionThreshold) {
			reset();
			return true;
		}

		return false;
	}

	private double euclidianNorm(double[] vector) {
		double sum = 0;
		for (int i = 0; i < vector.length; i++) {
			sum += vector[i] * vector[i];
		}
		return Math.sqrt(sum);
	}

	private void reset() {
		increasingMT = Double.MAX_VALUE;
		decreasingMT = Double.MIN_VALUE;
		// private double checkCounter = 0;
		for (int i = 0; i < numberOfDimensions; i++) {
			overallWeightedSum[i] = 0;
		}
		overallTotalWeight = 0;
	}

}

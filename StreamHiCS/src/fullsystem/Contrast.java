package fullsystem;

import org.apache.commons.math3.util.MathArrays;

import statisticaltests.KolmogorovSmirnov;
import statisticaltests.StatisticalTest;
import streamdatastructures.DataBundle;
import streamdatastructures.SummarisationAdapter;
import subspace.Subspace;
import weka.core.Instance;

/**
 * This is a super class for the contrast calculation.
 * 
 * @author Vincent
 *
 */
public class Contrast {

	/**
	 * Number of Monte Carlo iterations in the contrast evaluation. m must be
	 * positive.
	 */
	private int m;

	/**
	 * The relative size of the conditional (sliced) sample in relation to the
	 * whole data set. The number must be positive.
	 */
	private double alpha;

	/**
	 * The {@link StatisticalTest} used to calculate the deviation of the
	 * marginal sample and the conditional (sliced) sample.
	 */
	private StatisticalTest statisticalTest;

	/**
	 * The {@link SummarisationAdapter} providing access to the data stream
	 * summarisation structure.
	 */
	private SummarisationAdapter summarisationAdapter;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param m
	 *            The number of Monte Carlo iterations for the estimation of the
	 *            conditional density
	 * @param alpha
	 *            The fraction of data that should be selected in the estimation
	 *            of the conditional density
	 * @param summarisationAdapter
	 *            The {@link SummarisationAdapter} providing access to the data
	 *            stream summarisation structure
	 */
	public Contrast(int m, double alpha, SummarisationAdapter summarisationAdapter) {
		this.m = m;
		this.alpha = alpha;
		this.statisticalTest = new KolmogorovSmirnov();
		this.summarisationAdapter = summarisationAdapter;
	}

	/**
	 * Adds an @link{Instance} to the {@link SummarisationAdaper}.
	 * 
	 * @param instance
	 *            The @link{Instance} to be added.
	 */
	public void add(Instance instance) {
		summarisationAdapter.add(instance);
	}

	/**
	 * Clears all stored {@link Instance}s.
	 */
	public void clear() {
		summarisationAdapter.clear();
	}

	/**
	 * Returns the number of elements in the {@link SummarisationAdapter}.
	 * 
	 * @return the number of elements in the {@link SummarisationAdapter}.
	 */
	public int getNumberOfElements() {
		return summarisationAdapter.getNumberOfElements();
	}

	/**
	 * Calculates the contrast of the given @link{Subspace} by slicing the
	 * subspace and compairing the mariginal and slice samples using the
	 * {@link StatisticalTest}. This process is carried out m times and the
	 * average is returned.
	 * 
	 * @param subspace
	 *            The @link{Subspace} the contrast is calculated of.
	 * @return The contrast of the given @link{Subspace}.
	 */
	public double evaluateSubspaceContrast(Subspace subspace) {
		// Variable for collecting the intermediate results of the iterations
		double sum = 0;
		// A deviation could be NaN, so we wont count that calculation
		int numberOfCorrectTests = 0;
		// Calculate the fraction of instances selected per dimension
		double selectionAlpha = Math.pow(alpha, 1.0 / (subspace.size() - 1));
		int[] shuffledDimensions;
		DataBundle projectedData;
		DataBundle slicedData;
		double deviation;
		// Do Monte Carlo iterations
		int sumSliceWeight = 0;
		for (int i = 0; i < m; i++) {
			shuffledDimensions = subspace.getDimensions();
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// Get the projected data
			projectedData = summarisationAdapter.getProjectedData(shuffledDimensions[shuffledDimensions.length - 1]);
			// Get the randomly sliced data
			slicedData = summarisationAdapter.getSlicedData(shuffledDimensions, selectionAlpha);
			double[] weights = slicedData.getWeights();
			for(int j = 0; j < weights.length; j++){
				sumSliceWeight += weights[j];
			}
			if (slicedData.size() > 1) {
				// Calculate the deviation and add it to the overall sum
				deviation = statisticalTest.calculateWeightedDeviation(projectedData, slicedData);
				// deviation =
				// statisticalTest.calculateDeviation(projectedData.getData(),
				// slicedData.getData());
				if (!Double.isNaN(deviation)) {
					sum += deviation;
					numberOfCorrectTests++;
				}
			} else {
				// System.out.println("Slice too small: " + slicedData.size());
			}
		}
		System.out.println(((double)sumSliceWeight)/m);
		
		// Return the mean of the intermediate results. If all results were NaN,
		// 0 is returned.
		double mean = 0;
		if (numberOfCorrectTests > 0) {
			mean = sum / numberOfCorrectTests;
		}
		return mean;
	}
}

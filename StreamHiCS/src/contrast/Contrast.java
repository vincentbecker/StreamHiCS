package contrast;

import org.apache.commons.math3.util.MathArrays;

import statisticalTests.KolmogorovSmirnov;
import statisticalTests.StatisticalTest;
import streamDataStructures.Subspace;
import weka.core.Instance;

/**
 * This is a super class for the contrast calculation.
 * 
 * @author Vincent
 *
 */
public abstract class Contrast implements Callback {

	/**
	 * The @link{Callback} to notify on changes.
	 */
	private Callback callback;
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
	 * 
	 * 
	 * @param callback
	 * @param m
	 *            The number of Monte Carlo iterations for the estimation of the
	 *            conditional density.
	 * @param alpha
	 *            The fraction of data that should be selected in the estimation
	 *            of the conditional density.
	 * @param statisticalTest
	 */
	public Contrast(Callback callback, int m, double alpha) {
		this.callback = callback;
		this.m = m;
		this.alpha = alpha;
		this.statisticalTest = new KolmogorovSmirnov();
	}
	
	public void setCallback(Callback callback){
		this.callback = callback;
	}

	/**
	 * Add an @link{Instance}.
	 * 
	 * @param instance
	 *            The @link{Instance} to be added.
	 */
	public abstract void add(Instance instance);

	/**
	 * Clears all stored {@link Instance}s.
	 */
	public abstract void clear();
	
	public abstract int getNumberOfElements();

	/**
	 * Returns the data contained projected to the given reference dimension. 
	 * 
	 * @param referenceDimension The dimension the data is projected to. 
	 * @return The data projected to teh reference dimension. 
	 */
	public abstract double[] getProjectedData(int referenceDimension);

	/**
	 * Returns the one dimensional data of a random conditional sample
	 * corresponding to the last dimension in the int[] and the {@link Subspace}
	 * which contains this dimension. On every dimension in the {@link Subspace}
	 * except the specified one random range selections on instances (of the
	 * specified selection size) are done, representing a conditional sample for
	 * the given dimension.
	 * 
	 * @param shuffledDimensions
	 *            The dimensions. The last one is the one for which a random
	 *            conditional sample should be drawn.
	 * @param selectionAlpha
	 *            The fraction of instances that should be selected per
	 *            dimension (i.e. the number of selected instances becomes
	 *            smaller per selection step).
	 * @return A double[] containing a random conditional sample corresponding
	 *         to the given dimension.
	 */
	public abstract double[] getSlicedData(int[] shuffledDimensions, double selectionAlpha);

	/**
	 * Calculates the contrast of the given @link{Subspace}.
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
		double deviation;
		// Do Monte Carlo iterations
		for (int i = 0; i < m; i++) {
			shuffledDimensions = subspace.getDimensions();
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// Get the projected data statistics
			double[] projectedData = getProjectedData(shuffledDimensions[shuffledDimensions.length - 1]);
			// Get the randomly sliced data statistics
			double[] slicedData = getSlicedData(shuffledDimensions, selectionAlpha);
			// Calculate the deviation and add it to the overall sum
			deviation = statisticalTest.calculateDeviation(projectedData, slicedData);
			if (!Double.isNaN(deviation)) {
				sum += deviation;
				numberOfCorrectTests++;
			}
		}

		// Return the mean of the intermediate results. If all results were NaN,
		// 0 is returned.
		double mean = 0;
		if (numberOfCorrectTests > 0) {
			mean = sum / numberOfCorrectTests;
		}
		return mean;
	}

	@Override
	public void onAlarm() {
		callback.onAlarm();
	}

}

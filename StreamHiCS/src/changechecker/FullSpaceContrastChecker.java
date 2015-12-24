package changechecker;

import fullsystem.Contrast;
import subspace.Subspace;

/**
 * This class represents a {@link ChangeChecker} that calculates the contrast of
 * the full space to check for a change in the distribution. It compares the
 * current contrast to a weighted average of the contrast.
 * 
 * @author Vincent
 *
 */
public class FullSpaceContrastChecker extends ChangeChecker {

	/**
	 * The {@link Contrast} instance..
	 */
	private Contrast contrastEvaluator;

	/**
	 * The full space.
	 */
	private Subspace fullSpace;

	/**
	 * The minimal contrast found.
	 */
	private double minContrast = Double.MIN_VALUE;

	/**
	 * The maximum contrast found.
	 */
	private double maxContrast = Double.MIN_VALUE;

	/**
	 * The weighted average of the contrast.
	 */
	private double weightedAverage;

	/**
	 * The factor how fast the weighted average adapts to new values.
	 */
	private double gamma;

	/**
	 * The threshold for the difference between current contrast and weighted
	 * average.
	 */
	private double threshold;

	/**
	 * Initialisation flag.
	 */
	private boolean init = false;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param checkInterval
	 *            The time interval when to check for a change
	 * @param numberOfDimensions
	 *            The number of dimensions
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance.
	 * @param gamma
	 *            The adaptation factor for the weighted average
	 * @param threshold
	 *            The threshold to detect a change
	 */
	public FullSpaceContrastChecker(int checkInterval, int numberOfDimensions, Contrast contrastEvaluator, double gamma,
			double threshold) {
		super(checkInterval);
		int[] dimensions = new int[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			dimensions[i] = i;
		}
		fullSpace = new Subspace(dimensions);
		this.contrastEvaluator = contrastEvaluator;
		this.gamma = gamma;
		this.threshold = threshold;
	}

	/**
	 * Sets the {@link Contrast} instance.
	 * 
	 * @param contrastEvaluator
	 *            The contrast claculator
	 */
	public void setContrastEvaluator(Contrast contrastEvaluator) {
		this.contrastEvaluator = contrastEvaluator;
	}

	@Override
	public boolean checkForChange() {
		return weightedAverageMethod();
	}

	/**
	 * The method to check for a change here is to continuously update a minimum
	 * and maximum of the contrast. As soon as the contrast differs more than
	 * the threshold in either direction, a change is detected.
	 * 
	 * @return True, if a change is detected, false otherwise.
	 */
	private boolean minMaxMethod() {
		double contrast = contrastEvaluator.evaluateSubspaceContrast(fullSpace);

		System.out.println("Contrast: " + contrast);

		if (!init) {
			minContrast = contrast;
			maxContrast = contrast;
			init = true;
		}

		double minDifference = minContrast - contrast;
		double maxDifference = contrast - maxContrast;

		if (contrast < minContrast) {
			minContrast = contrast;
		}
		if (contrast > maxContrast) {
			maxContrast = contrast;
		}

		// double difference = Math.abs(lastContrast - contrast);
		// lastContrast = contrast;
		System.out.println("MinDifference: " + minDifference + ", MaxDifference: " + maxDifference);
		if (minDifference > threshold || maxDifference > threshold) {
			minContrast = 0;
			maxContrast = 0;
			init = false;
			return true;
		}
		return false;
	}

	/**
	 * Maintains a weighted average of the contrast. As soon as the current
	 * contrast of the full space differs from the average more than the
	 * threshold, a change is detected.
	 * 
	 * @return True, if a change is detected, false otherwise.
	 */
	private boolean weightedAverageMethod() {
		double contrast = contrastEvaluator.evaluateSubspaceContrast(fullSpace);
		System.out.println("Contrast: " + contrast);

		double difference = contrast - weightedAverage;
		System.out.println("Difference to average: " + difference);

		if (Math.abs(difference) > threshold) {
			weightedAverage = contrast;
			return true;
		}

		weightedAverage = gamma * contrast + (1 - gamma) * weightedAverage;

		return false;
	}

}

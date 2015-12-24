package statisticaltests;

/**
 * This class represents a data structure to hold the mean, variance and weight
 * of a sample.
 * 
 * @author Vincent
 *
 */
public class StatisticsBundle {

	/**
	 * The mean of the sample.
	 */
	private double mean;

	/**
	 * The variance of the sample.
	 */
	private double variance;

	/**
	 * The weight of the sample.
	 */
	private double weight;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param mean
	 *            The mean of the sample.
	 * @param variance
	 *            The variance of the sample.
	 * @param weight
	 *            The weight of the sample.
	 */
	public StatisticsBundle(double mean, double variance, double weight) {
		this.mean = mean;
		this.variance = variance;
		this.weight = weight;
	}

	/**
	 * Returns the mean.
	 * 
	 * @return The mean.
	 */
	public double getMean() {
		return mean;
	}

	/**
	 * Returns the variance.
	 * 
	 * @return The variance.
	 */
	public double getVariance() {
		return variance;
	}

	/**
	 * Returns the weight.
	 * 
	 * @return The weight.
	 */
	public double getWeight() {
		return weight;
	}
}

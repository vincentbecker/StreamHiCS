package statisticalTests;

public class StatisticsBundle {
	private double mean;
	private double variance;
	private double weight;

	public StatisticsBundle(double mean, double variance, double weight) {
		this.mean = mean;
		this.variance = variance;
		this.weight = weight;
	}

	public double getMean() {
		return mean;
	}

	public double getVariance() {
		return variance;
	}

	public double getWeight() {
		return weight;
	}
}

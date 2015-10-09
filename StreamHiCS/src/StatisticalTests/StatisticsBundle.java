package statisticalTests;

public class StatisticsBundle {
	private double mean;
	private double variance;

	public StatisticsBundle(double mean, double variance) {
		this.mean = mean;
		this.variance = variance;
	}

	public double getMean() {
		return mean;
	}

	public double getVariance() {
		return variance;
	}
}

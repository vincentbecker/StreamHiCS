package statisticalTests;

public abstract class StatisticalTest {
	public abstract double calculateDeviation(double[] sample1, double[] sample2);
	public abstract double calculateDeviation(StatisticsBundle sb1, StatisticsBundle sb2);
}

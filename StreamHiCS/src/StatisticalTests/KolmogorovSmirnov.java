package statisticalTests;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KolmogorovSmirnov extends StatisticalTest {
	private KolmogorovSmirnovTest kolmogorovSmirnovTest;

	public KolmogorovSmirnov() {
		 kolmogorovSmirnovTest = new KolmogorovSmirnovTest();
	}

	@Override
	public double calculateDeviation(double[] sample1, double[] sample2) {
		double d = kolmogorovSmirnovTest.kolmogorovSmirnovStatistic(sample1, sample2);
		return d;
	}
}

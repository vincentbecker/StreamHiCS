package statisticalTests;

import contrast.DataBundle;

public class KolmogorovSmirnov extends StatisticalTest {
	private KolmogorovSmirnovTest kolmogorovSmirnovTest;

	public KolmogorovSmirnov() {
		kolmogorovSmirnovTest = new KolmogorovSmirnovTest();
	}

	@Override
	public double calculateDeviation(double[] sample1, double[] sample2) {
		if (sample1.length == 0 && sample2.length == 0) {
			return 0;
		}

		// Check if all values are the same, special case for KS test
		boolean same1 = true;
		for (int i = 0; i < sample1.length; i++) {
			if (sample1[i] != sample1[0]) {
				same1 = false;
				break;
			}
		}
		boolean same2 = true;
		for (int i = 0; i < sample2.length; i++) {
			if (sample2[i] != sample2[0]) {
				same2 = false;
				break;
			}
		}
		if (same1 && same2 && sample1[0] == sample2[0]) {
			return 0;
		}

		return kolmogorovSmirnovTest.kolmogorovSmirnovStatistic(sample1, sample2);
	}

	@Override
	public double calculateWeightedDeviation(DataBundle dataBundle1, DataBundle dataBundle2) {
		double[] sample1 = dataBundle1.getData();
		double[] weights1 = dataBundle1.getWeights();
		double[] sample2 = dataBundle2.getData();
		double[] weights2 = dataBundle2.getWeights();

		// Check if all values are the same, special case for KS test
		boolean same1 = true;
		for (int i = 0; i < sample1.length; i++) {
			if (sample1[i] != sample1[0]) {
				same1 = false;
				break;
			}
		}
		boolean same2 = true;
		for (int i = 0; i < sample2.length; i++) {
			if (sample2[i] != sample2[0]) {
				same2 = false;
				break;
			}
		}
		if (same1 && same2 && sample1[0] == sample2[0]) {
			return 0;
		}

		return kolmogorovSmirnovTest.weightedKolmogorovSmirnovStatistic(sample1, weights1, sample2, weights2);
	}
}

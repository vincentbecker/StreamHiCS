package statisticalTests;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class KolmogorovSmirnov extends StatisticalTest {
	private KolmogorovSmirnovTest kolmogorovSmirnovTest;
	//private double dMin;
	//private double dMax;

	public KolmogorovSmirnov() {
		kolmogorovSmirnovTest = new KolmogorovSmirnovTest();
		/*
		int expectedSampleSize = (int) alpha * numberOfInstances;
		dMin = calculateExpectedDMin(numberOfInstances, expectedSampleSize, 100);
		dMax = calculateExpectedDMax(numberOfInstances, expectedSampleSize);
		*/
	}

	/*
	private double calculateExpectedDMin(int numberOfInstances, int expectedSampleSize, int iterations) {
		double totalDev = 0.0;
		var selection = newIndexSelection(N)
		for(int i = 0; i < iterations; i++){
			selection.selectRandomly(M)
			totalDev += computeDeviationFromSelfSelection(selection);
		}
		return totalDev / iterations;
	}

	private double calculateExpectedDMax(int numberOfInstances, int expectedSampleSize) {
		double totalDeviation = 0.0;
		var selection = newIndexSelection(N)
		let possibleOffsets = selection.possibleOffsets(M)
		for offset in possibleOffsets.min .. possibleOffsets.max:
				    selection.selectBlock(M, offset)
				    totalDeviation += computeDeviationFromSelfSelection(selection)
		return totalDeviation / possibleOffsets.numPossible
	}
	*/

	@Override
	public double calculateDeviation(double[] sample1, double[] sample2) {
		double d = kolmogorovSmirnovTest.kolmogorovSmirnovStatistic(sample1, sample2);
		return d;
		//return (d - dMin) / (dMax - dMin);
		// double p = kolmogorovSmirnovTest.kolmogorovSmirnovTest(sample1,
		// sample2);
		// return 1 - p;
	}

	@Override
	public double calculateDeviation(StatisticsBundle sb1, StatisticsBundle sb2) {
		// TODO Auto-generated method stub
		return 0;
	}
}

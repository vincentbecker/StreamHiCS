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
		//Check if all values are the same, special case for KS test
		boolean same1 = true;
		for(int i = 0; i < sample1.length; i++){
			if(sample1[i] != sample1[0]){
				same1 = false;
				break;
			}
		}
		boolean same2 = true;
		for(int i = 0; i < sample2.length; i++){
			if(sample2[i] != sample2[0]){
				same2 = false;
				break;
			}
		}
		if(same1 && same2 && sample1[0] == sample2[0]){
			return 0;
		}
		
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

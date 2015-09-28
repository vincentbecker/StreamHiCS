package StatisticalTests;

import org.apache.commons.math3.stat.inference.TTest;

public class WelchTtest extends StatisticalTest {

	private TTest tTest;
	
	public WelchTtest(){
		tTest = new TTest();
	}
	
	@Override
	public double calculateDeviation(double[] sample1, double[] sample2) {
		double p = tTest.tTest(sample1, sample2);
		return 1 - p;
	}

}

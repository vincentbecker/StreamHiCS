package statisticaltests;

import streamdatastructures.DataBundle;

public abstract class StatisticalTest {
	public abstract double calculateDeviation(double[] sample1, double[] sample2);
	public abstract double calculateWeightedDeviation(DataBundle dataBundle1, DataBundle dataBundle2);
}

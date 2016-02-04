package environment;

import java.util.ArrayList;

public class AccuracyEvaluator {

	private ArrayList<Integer> classLabels;
	private ArrayList<Integer> predictions;

	public AccuracyEvaluator() {
		classLabels = new ArrayList<Integer>();
		predictions = new ArrayList<Integer>();
	}
	
	public void addPrediction(int predictedClass){
		predictions.add(predictedClass);
	}
	
	public void addClassLabel(int label){
		classLabels.add(label);
	}

	public double calculateOverallErrorRate() {
		int l = classLabels.size();
		assert (l == predictions.size());
		double numErrors = 0;
		for (int i = 0; i < l; i++) {
			if (classLabels.get(i) != predictions.get(i)) {
				numErrors++;
			}
		}
		return numErrors / l;
	}

	public double[] calculateSmoothedErrorRates(int smoothingInterval) {
		int l = classLabels.size();
		double[] smoothedErrorRates = new double[l];
		assert (l == predictions.size());
		double numErrors = 0;
		for (int i = 0; i < l; i++) {
			numErrors = 0;
			int j = i - smoothingInterval + 1;
			if (j < 0) {
				j = 0;
			}
			for (; j <= i; j++) {
				if (classLabels.get(j) != predictions.get(j)) {
					numErrors++;
				}
			}
			smoothedErrorRates[i] = numErrors / smoothingInterval;
		}
		return smoothedErrorRates;
	}
	
	public int size(){
		return classLabels.size();
	}
}

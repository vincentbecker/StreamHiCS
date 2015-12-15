package streamdatastructures;

import org.apache.commons.math3.util.MathArrays;

public class DataBundle {
	private double[] indexes;
	private double[] data;
	private double[] weights;

	public DataBundle(double[] data, double[] weights) {
		int n = data.length;
		if (n != weights.length) {
			throw new IllegalArgumentException("Data and weights have different length.");
		}
		this.data = data;
		this.weights = weights;
		this.indexes = new double[n];
		for(int i = 0; i < n; i++){
			indexes[i] = i;
		}
	}
	
	public double[] getIndexes(){
		return indexes;
	}
	
	public double[] getData() {
		return data;
	}

	public double[] getWeights() {
		return weights;
	}
	
	public boolean isEmpty(){
		if(data == null || data.length == 0){
			return true;
		}
		return false;
	}

	public void sort() {
		MathArrays.sortInPlace(data, indexes, weights);
	}
}

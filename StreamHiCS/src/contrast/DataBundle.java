package contrast;

public class DataBundle {
	private double[] data;
	private double[] weights;

	public DataBundle(double[] data, double[] weights) {
		if (data.length != weights.length) {
			throw new IllegalArgumentException("Data and weights have different length.");
		}
		this.data = data;
		this.weights = weights;
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
}

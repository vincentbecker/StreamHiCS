package centroids;

public class Centroid {
	private double[] vector;
	private int count = 0;
	
	public Centroid(double[] vector){
		this.vector = vector;
	}
	
	public double[] getVector() {
		return vector;
	}

	public void setVector(double[] vector) {
		this.vector = vector;
	}

	public int getCount() {
		return count;
	}
	
	public void setCount(int count) {
		this.count = count;
	}

	public void increment() {
		this.count++;
	}
}

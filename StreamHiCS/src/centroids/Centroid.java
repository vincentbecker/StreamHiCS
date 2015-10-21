package centroids;

public class Centroid {
	private final long id;
	private double[] vector;
	private int count = 0;
	private double weight = 0;
	private int lastUpdate;
	private double fadingFactor;

	public Centroid(long id, double[] vector, double fadingFactor, int lastUpdate) {
		this.id = id;
		this.vector = vector;
		this.fadingFactor = fadingFactor;
		this.lastUpdate = lastUpdate;
	}

	public long getId() {
		return id;
	}

	public double[] getVector() {
		return vector;
	}

	public void setVector(double[] vector) {
		this.vector = vector;
	}

	public double getWeight() {
		return this.weight;
	}

	public int getCount() {
		return count;
	}

	public void increment() {
		weight++;
		count++;
	}

	public void fade(int currentTime) {
		weight = weight * Math.pow(fadingFactor, currentTime - lastUpdate);
		lastUpdate = currentTime;
	}
	
	public double euclideanDistance(double[] vector) {
		double[] v1 = this.vector;
		double[] v2 = vector;
		if (v1.length != v2.length) {
			throw new IllegalArgumentException("Centroid vectors are of different length.");
		}
		double distance = 0;
		for (int i = 0; i < v1.length; i++) {
			distance += Math.pow(v1[i] - v2[i], 2);
		}

		return Math.sqrt(distance);
	}


	/**
	 * Returns a string representation of this object.
	 * 
	 * @return A string representation of this object.
	 */
	public String toString() {
		String s = "ID: " + id + " [";
		for (int i = 0; i < vector.length - 1; i++) {
			s += vector[i] + ", ";
		}
		s += (vector[vector.length - 1] + "]");
		return s + " Weight: " + weight;
	}
}

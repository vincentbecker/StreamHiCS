package centroids;

public abstract class Centroid {
	protected int lastUpdate;
	protected double weight = 1;
	protected double negLambda;
	
	public Centroid(double negLambda, int currentTime){
		this.negLambda = negLambda;
		this.lastUpdate = currentTime;
	}
	
	public abstract double[] getCentre();
	public boolean addPoint(double[] point, int currentTime){
		fade(currentTime);
		boolean added = addPointImpl(point);
		if(added){
			weight++;
		}
		return added;
	}
	public abstract boolean addPointImpl(double[] point);
	public double getRadius(int currentTime){
		fade(currentTime);
		return getRadiusImpl();
	}
	public abstract double getRadiusImpl();
	
	public double getWeight(int currentTime){
		if(currentTime >= 0){
			fade(currentTime);
		}
		return weight;
	}
	
	private void fade(int currentTime) {
		if(lastUpdate != currentTime){
			weight = weight * Math.pow(2.0, negLambda * (currentTime - lastUpdate));
			fadeImpl(currentTime);
			lastUpdate = currentTime;
		}
	}
	
	public abstract void fadeImpl(int currentTime);
	
	public double euclideanDistance(double[] vector) {
		double[] v1 = getCentre();
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
		String s = "[";
		double[] vector = getCentre();
		for (int i = 0; i < vector.length - 1; i++) {
			s += vector[i] + ", ";
		}
		s += (vector[vector.length - 1] + "]");
		return s + " Weight: " + weight;
	}
}

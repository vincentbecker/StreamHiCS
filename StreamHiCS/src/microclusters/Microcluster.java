package microclusters;

/**
 * An abstract class for a micro-clustering approach with fading micro-clusters.
 * The fading procedure is implemented by exponential fading. The subclasses do
 * not have to take care of the fading.
 * 
 * @author Vincent
 *
 */
public abstract class Microcluster {
	/**
	 * The time of the last update. Used to calculate how much to fade this
	 * {@link Microcluster}.
	 */
	protected int lastUpdate;

	/**
	 * The weight. Initially 1.
	 */
	protected double weight = 1;

	/**
	 * The negative lambda for fading.
	 */
	protected double negLambda;

	/**
	 * Sets the attributes.
	 * 
	 * @param negLambda
	 *            The negative lambda for fading
	 * @param currentTime
	 *            The current time of creation
	 */
	public Microcluster(double negLambda, int currentTime) {
		this.negLambda = negLambda;
		this.lastUpdate = currentTime;
	}

	/**
	 * Returns the centre of the {@link Microcluster}.
	 * 
	 * @return The centre of the {@link Microcluster}.
	 */
	public abstract double[] getCentre();

	/**
	 * Adds a point to the {@link Microcluster}.
	 * 
	 * @param point
	 *            The point to add
	 * @param currentTime
	 *            The current time
	 * @return True, if the point could be added to the {@link Microcluster}, false
	 *         otherwise.
	 */
	public boolean addPoint(double[] point, int currentTime) {
		fade(currentTime);
		return addPointImpl(point);
	}

	/**
	 * The implementation of the adding procedure of the subclass.
	 * 
	 * @param point
	 *            The point to be added.
	 * 
	 * @return True, if the point could be added to the {@link Microcluster}, false
	 *         otherwise.
	 */
	public abstract boolean addPointImpl(double[] point);

	/**
	 * Returns the radius of the {@link Microcluster}. The {@link Microcluster} is faded before
	 * calculating the radius.
	 * 
	 * @param currentTime
	 *            The current time
	 * 
	 * @return The radius of the {@link Microcluster}.
	 */
	public double getRadius(int currentTime) {
		fade(currentTime);
		return getRadiusImpl();
	}

	/**
	 * The implementation of the radius calculation of the subclass.
	 * 
	 * @return The radius of the {@link Microcluster}.
	 */
	public abstract double getRadiusImpl();

	/**
	 * Returns the weight of the {@link Microcluster}.
	 * 
	 * @param currentTime
	 *            The current time
	 * @return If the time is 0 or higher, then the faded weight. Otherwise the
	 *         currently stored value is returned.
	 */
	public double getWeight(int currentTime) {
		if (currentTime >= 0) {
			fade(currentTime);
		}
		return weight;
	}

	/**
	 * Fades the @link Microcluster} using exponential fading. Here the fading of
	 * the weight is taken care of, the subclasses take care of the fading of
	 * the other cluster features.
	 * 
	 * @param currentTime
	 *            The current time
	 */
	private void fade(int currentTime) {
		if (lastUpdate != currentTime) {
			weight = weight * Math.pow(2.0, negLambda * (currentTime - lastUpdate));
			fadeImpl(currentTime);
			lastUpdate = currentTime;
		}
	}

	/**
	 * The implementation of the fading procedure of the subclass.
	 * 
	 * @param currentTime
	 *            The current time
	 */
	public abstract void fadeImpl(int currentTime);

	/**
	 * Calculates the Euclidean distance to a given vector.
	 * 
	 * @param vector
	 *            The vector
	 * @return The distance of the {@link Microcluster} to the vector.
	 * @throws {@link
	 *             IllegalArgumentException} if the vector does not have the
	 *             same number of dimensions as the {@link Microcluster}.
	 */
	public double euclideanDistance(double[] vector) {
		double[] v1 = getCentre();
		double[] v2 = vector;
		if (v1.length != v2.length) {
			throw new IllegalArgumentException("Vectors are of different length.");
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

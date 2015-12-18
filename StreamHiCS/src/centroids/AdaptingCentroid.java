package centroids;

public class AdaptingCentroid extends Centroid {

	private double[] centre;
	private double learningRate;
	private double radius;

	public AdaptingCentroid(double[] vector, double negLambda, int currentTime, double radius, double learningRate) {
		super(negLambda, currentTime);
		this.centre = vector;
		this.radius = radius;
		this.learningRate = learningRate;
	}

	@Override
	public double[] getCentre() {
		return centre;
	}

	@Override
	public boolean addPointImpl(double[] point) {

		double distance = this.euclideanDistance(point);

		if (distance > radius) {
			return false;
		}

		for (int i = 0; i < centre.length; i++) {
			centre[i] += point[i];
		}

		// Adapt according to the weight of the centroid
		double adaptationRate = h(point) / (weight + 1);
		for (int i = 0; i < centre.length; i++) {
			centre[i] += adaptationRate * (point[i] - centre[i]);
		}

		return true;
	}

	@Override
	public void fadeImpl(int currentTime) {
		// No special fading
	}

	@Override
	public double getRadiusImpl() {
		return radius;
	}

	/**
	 * Calculates a factor how strongly a {@link AdaptingCentroid} should be
	 * adapted to a vector.
	 * 
	 * @param c
	 *            The {@link AdaptingCentroid}.
	 * @param vector
	 *            The input vector.
	 * @return The factor how strongly the given {@link AdaptingCentroid} should
	 *         be adapted to the input vector.
	 */
	private double h(double[] vector) {
		return learningRate;
	}
}

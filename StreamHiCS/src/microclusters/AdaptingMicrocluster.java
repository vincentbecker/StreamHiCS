package microclusters;

import weka.core.Instance;

/**
 * This class implements a micro-clustering approach where the {@link Microcluster}s
 * adapt to the incoming {@link Instance}s by moving towards them. Here, the
 * radius is fixed.
 * 
 * @author Vincent
 *
 */
public class AdaptingMicrocluster extends Microcluster {

	/**
	 * The centre.
	 */
	private double[] centre;

	/**
	 * The learning rate for the adaptation of the {@link AdaptingMicrocluster}.
	 */
	private double learningRate;

	/**
	 * The radius of the {@link AdaptingMicrocluster}.
	 */
	private double radius;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param vector
	 *            The initial centre of the {@ AdaptingMicrocluster}
	 * @param negLambda
	 *            The negative lambda value for the fading procedure
	 * @param currentTime
	 *            The current time at which this {@ AdaptingMicrocluster} is created
	 * @param radius
	 *            The radius of this {@link AdaptingMicrocluster}
	 * @param learningRate
	 *            The learning rate for the adaptation of the
	 *            {@link AdaptingMicrocluster}
	 */
	public AdaptingMicrocluster(double[] vector, double negLambda, int currentTime, double radius, double learningRate) {
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

		/*
		for (int i = 0; i < centre.length; i++) {
			centre[i] += point[i];
		}
		*/

		weight++;

		// Adapt according to the weight of the micro-cluster
		double adaptationRate = h(point) / (weight);
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
	 * Calculates a factor how strongly a {@link AdaptingMicrocluster} should be
	 * adapted to a vector.
	 * 
	 * @param vector
	 *            The input vector.
	 * @return The factor how strongly the given {@link AdaptingMicrocluster} should
	 *         be adapted to the input vector.
	 */
	private double h(double[] vector) {
		// Here simply use the learning rate by itself
		return learningRate;
	}
}

package centroids;

import java.util.ArrayList;

import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.StringOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;

/**
 * This class represents a {@link AdaptingCentroid} implementation which adapts
 * to the input data by moving the {@link Centroid}s in direction of the
 * incoming {@link Instance}s.
 * 
 * @author Vincent
 *
 */
public class FadingCentroids extends AbstractOptionHandler {
	/**
	 * Serial version ID.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The centroids.
	 */
	private ArrayList<Centroid> centroids;

	/**
	 * The number of dimensions of the full space.
	 */
	private int numberOfDimensions = -1;

	/**
	 * The negative lambda for fading.
	 */
	private double negLambda;

	/**
	 * For each incoming instance we search for the nearest
	 * {@link AdaptingCentroid} in which radius the instance falls.
	 */
	private double radius;

	/**
	 * Keeps track of the current time.
	 */
	private int time = 0;

	/**
	 * The threshold determining when a {@link AdaptingCentroid} is removed.
	 */
	private final double weightThreshold = 0.25;

	/**
	 * A flag showing if all weights are updated currently.
	 */
	private boolean updated = true;

	/**
	 * The learning rate for the adaptation of the {@link AdaptingCentroid}.
	 */
	private double learningRate;

	/**
	 * A flag showing whether the number of dimensions was already set.
	 */
	private boolean initialized = false;

	/**
	 * Counts how many {@link Centroid}s faded away throughout the streaming
	 * process.
	 */
	public int faded = 0;

	/**
	 * Determines which version of the centroids to use.
	 */
	private String centroidVersion;

	/**
	 * The option determining the horizon for fading.
	 */
	public IntOption horizonOption = new IntOption("horizon", 'h', "Horizon", 1000, 1, Integer.MAX_VALUE);

	/**
	 * The option determining the radius of a {@link Centroid} determining if a
	 * new @link{Instance} could belong to it.
	 */
	public FloatOption radiusOption = new FloatOption("radius", 'r', "Radius", 1, 0, Double.MAX_VALUE);

	/**
	 * The option determining the learning rate for the adaptation of the
	 * {@link Centroid}s.
	 */
	public FloatOption learningRateOption = new FloatOption("scale", 's', "Scale.", 1, 0, Double.MAX_VALUE);

	public StringOption centroidVersionOption = new StringOption("centroidVersion", 'c', "Select the centroid version",
			"adapting");

	/**
	 * An array containing all the currently held {@link Centroid}s.
	 * 
	 * @return An array of the Centroid}s.
	 */
	public Centroid[] getCentroids() {
		updateWeights();
		Centroid[] cs = new Centroid[centroids.size()];
		centroids.toArray(cs);
		return cs;
	}

	/**
	 * Adds an incoming {@link Instance} to the nearest {@link Centroid} (by
	 * euclidean distance), if possible. If not, a new centroid is added, of
	 * which the initial centre is set to the instance.
	 * 
	 * @param instance
	 */
	public void add(Instance instance) {
		if (!initialized) {
			this.numberOfDimensions = instance.numAttributes();
			initialized = true;
		} else {
			if (instance.numAttributes() != numberOfDimensions) {
				throw new IllegalArgumentException("Instance has wrong number of dimensions. ");
			}
		}
		// Create vector
		double[] vector = instance.toDoubleArray();

		// Find the closest centroid
		Centroid nearest = findNearestCentroid(vector);
		if (nearest == null) {
			createCentroid(vector);
		} else {
			if (!nearest.addPoint(vector, time)) {
				createCentroid(vector);
			}
		}
		time++;
		updated = false;
	}

	/**
	 * Creates a new {@link Centroid} with the given vector as the initial
	 * centre and adds it to the centorid collection.
	 * 
	 * @param vector
	 *            The vector
	 */
	private void createCentroid(double[] vector) {
		Centroid c;
		switch (centroidVersion) {
		case "adapting":
			c = new AdaptingCentroid(vector, negLambda, time, radius, learningRate);
			break;
		case "radius":
			c = new RadiusCentroid(vector, negLambda, time, radius);
			break;
		default:
			c = new AdaptingCentroid(vector, negLambda, time, radius, learningRate);
		}
		centroids.add(c);
	}

	/**
	 * Finds the nearest {@link Centroid} to the given vector. The vector has to
	 * fall in the radius of the {@link AdaptingCentroid}. Before searching for
	 * the nearest {@link AdaptingCentroid} the weights of all
	 * {@link AdaptingCentroid}s are updated and the centroids removed if their
	 * weight falls below the threshold.
	 * 
	 * @param vector
	 *            The input vector
	 * @return The nearest {@link AdaptingCentroid} to the given vector, in
	 *         which's radius the vector falls. Null, if such a vector does not
	 *         exist
	 */
	private Centroid findNearestCentroid(double[] vector) {
		double distance;
		double minDistance = Double.MAX_VALUE;
		Centroid nearestCentroid = null;

		updateWeights();

		for (Centroid c : centroids) {
			distance = c.euclideanDistance(vector);
			if (distance < minDistance) {
				minDistance = distance;
				nearestCentroid = c;
			}
		}
		return nearestCentroid;
	}

	/**
	 * Updates the weights of all contained {@link AdaptingCentroid}s. If the
	 * weight falls below the weight threshold the {@link AdaptingCentroid} is
	 * removed.
	 */
	private void updateWeights() {
		if (!updated) {
			ArrayList<Centroid> removalList = new ArrayList<Centroid>();
			for (Centroid c : centroids) {
				// Fading is already done in getWeight()
				// c.fade(time);
				if (c.getWeight(time) < weightThreshold) {
					removalList.add(c);
				}
			}

			faded += removalList.size();

			for (Centroid c : removalList) {
				centroids.remove(c);
			}
			updated = true;
		}
	}

	/**
	 * Resets the implementation.
	 */
	public void clear() {
		centroids.clear();
		time = 0;
		initialized = false;
		numberOfDimensions = -1;
	}

	/**
	 * Returns the number of currently held {@link Centroids}.
	 * 
	 * @return The number of currently held {@link Centroids}.
	 */
	public int getNumberOfInstances() {
		updateWeights();
		return centroids.size();
	}

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		this.centroids = new ArrayList<Centroid>();
		// Calculating the fading scale log2(threshold) / horizon. Here the log2
		// is -2. Beware, relies on weight threshold being 0.25
		this.negLambda = -2.0 / horizonOption.getValue();
		this.radius = radiusOption.getValue();
		this.learningRate = learningRateOption.getValue();
		this.centroidVersion = centroidVersionOption.getValue();
		initialized = false;
		time = 0;
	}
}
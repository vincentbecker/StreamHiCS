package microclusters;

import java.util.ArrayList;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.StringOption;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;
import weka.core.Instance;

/**
 * This class represents a micro-clustering implementation which adapts
 * to the input data by moving the {@link Microcluster}s in direction of the
 * incoming {@link Instance}s.
 * 
 * @author Vincent
 *
 */
public class FadingMicroclusters extends AbstractClusterer {
	/**
	 * Serial version ID.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * The {@link Microcluster}s.
	 */
	private ArrayList<Microcluster> microclusters;

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
	 * {@link Microcluster} in which radius the instance falls.
	 */
	private double radius;

	/**
	 * Keeps track of the current time.
	 */
	private int time = 0;

	/**
	 * The threshold determining when a {@link Microcluster} is removed.
	 */
	private final double weightThreshold = 0.25;

	/**
	 * A flag showing if all weights are updated currently.
	 */
	private boolean updated = true;

	/**
	 * The learning rate for the adaptation of the {@link Microcluster}.
	 */
	private double learningRate;

	/**
	 * A flag showing whether the number of dimensions was already set.
	 */
	private boolean initialized = false;

	/**
	 * Counts how many {@link Microcluster}s faded away throughout the streaming
	 * process.
	 */
	public int faded = 0;

	/**
	 * Determines which version of the {@link Microcluster}s to use.
	 */
	private String microclusterVersion;

	/**
	 * The option determining the horizon for fading.
	 */
	public IntOption horizonOption = new IntOption("horizon", 'h', "Horizon", 1000, 1, Integer.MAX_VALUE);

	/**
	 * The option determining the radius of a {@link Microcluster} determining if a
	 * new @link{Instance} could belong to it.
	 */
	public FloatOption radiusOption = new FloatOption("radius", 'r', "Radius", 1, 0, Double.MAX_VALUE);

	/**
	 * The option determining the learning rate for the adaptation of the
	 * {@link Microcluster}s.
	 */
	public FloatOption learningRateOption = new FloatOption("scale", 's', "Scale.", 1, 0, Double.MAX_VALUE);

	/**
	 * The option to decide which version of micro-clusters. Either adapting or radius. 
	 */
	public StringOption microclusterVersionOption = new StringOption("microclusterVersion", 'c', "Select the micro-cluster version",
			"adapting");

	/**
	 * An array containing all the currently held {@link Microcluster}s.
	 * 
	 * @return An array of the {@link Microcluster}s.
	 */
	public Microcluster[] getMicroclusters() {
		updateWeights();
		Microcluster[] cs = new Microcluster[microclusters.size()];
		microclusters.toArray(cs);
		return cs;
	}

	/**
	 * Adds an incoming {@link Instance} to the nearest {@link Microcluster} (by
	 * Euclidean distance), if possible. If not, a new {@link Microcluster} is added, of
	 * which the initial centre is set to the instance.
	 * 
	 * @param instance The instance
	 */
	@Override
	public void trainOnInstanceImpl(Instance instance) {
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

		// Find the closest micro-cluster
		Microcluster nearest = findNearestMicrocluster(vector);
		if (nearest == null) {
			createMicrocluster(vector);
		} else {
			if (!nearest.addPoint(vector, time)) {
				createMicrocluster(vector);
			}
		}
		time++;
		updated = false;
	}

	/**
	 * Creates a new {@link Microcluster} with the given vector as the initial
	 * centre and adds it to the Microcluster collection.
	 * 
	 * @param vector
	 *            The vector
	 */
	private void createMicrocluster(double[] vector) {
		Microcluster c;
		switch (microclusterVersion) {
		case "adapting":
			c = new AdaptingMicrocluster(vector, negLambda, time, radius, learningRate);
			break;
		case "radius":
			c = new RadiusMicrocluster(vector, negLambda, time, radius);
			break;
		default:
			System.out.println("Micro-cluster implementation not available!");
			c = null;
		}
		microclusters.add(c);
	}

	/**
	 * Finds the nearest {@link Microcluster} to the given vector. The vector has to
	 * fall in the radius of the {@link AdaptingMicrocluster}. Before searching for
	 * the nearest {@link AdaptingMicrocluster} the weights of all
	 * {@link AdaptingMicrocluster}s are updated and the micro-clusters removed if their
	 * weight falls below the threshold.
	 * 
	 * @param vector
	 *            The input vector
	 * @return The nearest {@link AdaptingMicrocluster} to the given vector, in
	 *         which's radius the vector falls. Null, if such a vector does not
	 *         exist
	 */
	private Microcluster findNearestMicrocluster(double[] vector) {
		double distance;
		double minDistance = Double.MAX_VALUE;
		Microcluster nearestMicrocluster = null;

		updateWeights();

		for (Microcluster c : microclusters) {
			distance = c.euclideanDistance(vector);
			if (distance < minDistance) {
				minDistance = distance;
				nearestMicrocluster = c;
			}
		}
		return nearestMicrocluster;
	}

	/**
	 * Updates the weights of all contained {@link AdaptingMicrocluster}s. If the
	 * weight falls below the weight threshold the {@link AdaptingMicrocluster} is
	 * removed.
	 */
	private void updateWeights() {
		if (!updated) {
			ArrayList<Microcluster> removalList = new ArrayList<Microcluster>();
			for (Microcluster c : microclusters) {
				// Fading is already done in getWeight()
				// c.fade(time);
				if (c.getWeight(time) < weightThreshold) {
					removalList.add(c);
				}
			}

			faded += removalList.size();

			for (Microcluster c : removalList) {
				microclusters.remove(c);
			}
			updated = true;
		}
	}

	/**
	 * Returns the number of currently held {@link Microcluster}s.
	 * 
	 * @return The number of currently held {@link Microcluster}s.
	 */
	public int getNumberOfInstances() {
		updateWeights();
		return microclusters.size();
	}

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		this.microclusters = new ArrayList<Microcluster>();
		// Calculating the fading scale log2(threshold) / horizon. Here the log2
		// is -2. Beware, relies on weight threshold being 0.25
		this.negLambda = -2.0 / horizonOption.getValue();
		this.radius = radiusOption.getValue();
		this.learningRate = learningRateOption.getValue();
		this.microclusterVersion = microclusterVersionOption.getValue();
		initialized = false;
		time = 0;
	}

	@Override
	public Clustering getClusteringResult() {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Clustering getMicroClusteringResult(){
		
		return clustering;
	}
	
	@Override
	public boolean implementsMicroClusterer() {
		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void resetLearningImpl() {
		microclusters.clear();
		time = 0;
		initialized = false;
		numberOfDimensions = -1;
	}
	
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}
}
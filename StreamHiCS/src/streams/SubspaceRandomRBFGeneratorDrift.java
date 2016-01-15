package streams;

import moa.options.IntOption;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.options.FlagOption;
import moa.options.FloatOption;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 * This class represents a stream generator very similar to the MOA
 * {@link RandomRBFGeneratorDrift}. The only difference is that a specific
 * number of centroids, generate the instances in subspaces, i.e. in all other
 * dimensions not contained in the subspace the attribute values are noise.
 * 
 * @author Vincent
 *
 */
public class SubspaceRandomRBFGeneratorDrift extends RandomRBFGeneratorDrift {

	@Override
	public String getPurposeString() {
		return "Generates a subspace random radial basis function stream with drift.";
	}

	private static final long serialVersionUID = 1L;

	private int numberDimensions;

	/**
	 * Determines whether each centroid uses the same subspace.
	 */
	public FlagOption sameSubspaceOption = new FlagOption("sameSubpace", 'b',
			"Determines if every subspace Cluster uses the same subspace.");

	/**
	 * Determines the average size of a subspace, i.e. the number of dimensions.
	 */
	public IntOption avgSubspaceSizeOption = new IntOption("avgSubspaceSize", 'p', "The average size of a subspace.",
			50, 0, Integer.MAX_VALUE);

	/**
	 * Determines whether the size of the subspaces is fixed or not.
	 */
	public FlagOption randomSubspaceSizeOption = new FlagOption("randomSubspaceSize", 'f',
			"Determines if the size of the subspaces is fixed or may vary.");

	/**
	 * Determines how many centroids use subspaces.
	 */
	public IntOption numSubspaceCentroidsOption = new IntOption("numSubspaceCentroids", 'e',
			"The number of centroids with a subspace.", 50, 0, Integer.MAX_VALUE);

	public FloatOption scaleIrrelevantDimensionsOption = new FloatOption("scaleIrrelevantDimensions", 'd',
			"The scale the uniformly random value (-1, 1) is scaled with for irrelevant dimensions, i.e. dimensions not contained in the subspace.",
			5, 0, Double.MAX_VALUE);

	/**
	 * The subspaces, indexed by the labels of the centroids, which use them.
	 */
	private Subspace[] subspaces;

	/**
	 * The scale applied to the uniform random value in the range (-1, 1) for
	 * irrelevant dimensions, i.e. dimensions which are not contained in the
	 * corresponding subspace.
	 */
	private double scaleIrrelevant;

	@Override
	public Instance nextInstance() {
		// Update Centroids with drift
		int len = this.numDriftCentroidsOption.getValue();
		if (len > this.centroids.length) {
			len = this.centroids.length;
		}
		for (int j = 0; j < len; j++) {
			for (int i = 0; i < this.numAttsOption.getValue(); i++) {
				this.centroids[j].centre[i] += this.speedCentroids[j][i] * this.speedChangeOption.getValue();
				if (this.centroids[j].centre[i] > 1) {
					this.centroids[j].centre[i] = 1;
					this.speedCentroids[j][i] = -this.speedCentroids[j][i];
				}
				if (this.centroids[j].centre[i] < 0) {
					this.centroids[j].centre[i] = 0;
					this.speedCentroids[j][i] = -this.speedCentroids[j][i];
				}
			}
		}

		// Create instance
		int centroidIndex = MiscUtils.chooseRandomIndexBasedOnWeights(this.centroidWeights, this.instanceRandom);
		Centroid centroid = this.centroids[centroidIndex];
		Subspace s = subspaces[centroidIndex];
		double[] attVals = new double[numberDimensions + 1];
		for (int i = 0; i < numberDimensions; i++) {
			attVals[i] = (this.instanceRandom.nextDouble() * 2.0) - 1.0;
		}
		double magnitude = 0.0;
		for (int i = 0; i < numberDimensions; i++) {
			magnitude += attVals[i] * attVals[i];
		}
		magnitude = Math.sqrt(magnitude);
		double desiredMag = this.instanceRandom.nextGaussian() * centroid.stdDev;
		double scale = desiredMag / magnitude;
		for (int i = 0; i < numberDimensions; i++) {
			if (s != null && !s.contains(i)) {
				attVals[i] = ((this.instanceRandom.nextDouble() * 2.0) - 1.0) * scaleIrrelevant;
			} else {
				attVals[i] = centroid.centre[i] + attVals[i] * scale;
			}
		}

		Instance inst = new DenseInstance(1.0, attVals);
		inst.setDataset(getHeader());
		inst.setClassValue(centroid.classLabel);

		return inst;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		super.prepareForUseImpl(monitor, repository);

		numberDimensions = numAttsOption.getValue();
		monitor.setCurrentActivity("Preparing subspace random RBF...", -1.0);
		// Get scale
		this.scaleIrrelevant = scaleIrrelevantDimensionsOption.getValue();
		// Generate subspaces
		subspaces = new Subspace[numCentroidsOption.getValue()];

		if (sameSubspaceOption.isSet()) {
			// Use the the same subspace for all centroids with subspaces
			Subspace s = createSubspace();
			for (int i = 0; i < subspaces.length; i++) {
				subspaces[i] = s;
			}
		} else {
			for (int i = 0; i < subspaces.length; i++) {
				subspaces[i] = createSubspace();
			}
		}

	}

	private Subspace createSubspace() {
		int numRelevantDims = 0;
		if (randomSubspaceSizeOption.isSet()) {
			numRelevantDims = (int) (avgSubspaceSizeOption.getValue() + (instanceRandom.nextBoolean() ? -1 : 1)
					* avgSubspaceSizeOption.getValue() * instanceRandom.nextDouble());
		} else {
			numRelevantDims = avgSubspaceSizeOption.getValue();
		}
		if (numRelevantDims < 2) {
			numRelevantDims = 2; // At least 2
		}
		if (numRelevantDims > numAttsOption.getValue()) {
			numRelevantDims = numberDimensions;
		}

		Subspace s = new Subspace();

		for (int j = 0; j < numRelevantDims; j++) {
			int relevantDim = (int) (instanceRandom.nextDouble() * (numAttsOption.getValue()));
			while (s.contains(relevantDim)) {
				relevantDim = (int) (instanceRandom.nextDouble() * (numAttsOption.getValue()));
			}
			s.addDimension(relevantDim);
		}
		return s;
	}

	public SubspaceSet getSubspaces() {
		SubspaceSet set = new SubspaceSet();
		for (int i = 0; i < subspaces.length; i++) {
			set.addSubspace(subspaces[i]);
		}
		return set;
	}
}

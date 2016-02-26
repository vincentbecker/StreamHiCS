package streams;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import moa.core.ObjectRepository;
import moa.options.FlagOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import weka.core.DenseInstance;
import weka.core.Instance;

public class DriftingGaussians2 extends SubspaceRandomRBFGeneratorDrift {

	/**
	 * Determines how many centroids use subspaces.
	 */
	public IntOption numSubspacesOption = new IntOption("numSubspaces", 'u', "The number of subspaces.", 1, 0,
			Integer.MAX_VALUE);

	/**
	 * Determines whether the size of the subspaces is fixed or not.
	 */
	public FlagOption disjointSubspacesOption = new FlagOption("disjointSubspaces", 'j',
			"Determines if the subspaces overlap or not.");

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -8561104512189724401L;

	/**
	 * The underlying stream generator.
	 */
	private MultivariateNormalDistribution currentNormalDistribution;

	/**
	 * The underlying stream generator.
	 */
	private MultivariateNormalDistribution newNormalDistribution;

	/**
	 * The covariance value every entry of the covariance matrix is set to
	 * except the diagonal.
	 */
	private final double covariance = 0.9;

	/**
	 * The current {@link Subspace} used.
	 */
	private Subspace currentSubspace;

	/**
	 * The new {@link Subspace} used on case of a virtual drift.
	 */
	private Subspace newSubspace;

	/**
	 * The number of subspaces.
	 */
	private int numberSubspaces;

	/**
	 * Whether the subspace overlap or not.
	 */
	private boolean disjointSubspaces;

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
		// int centroidIndex =
		// MiscUtils.chooseRandomIndexBasedOnWeights(this.centroidWeights,
		// this.instanceRandom);
		int centroidIndex = modelRandom.nextInt(numberCentroids);
		Centroid centroid = this.centroids[centroidIndex];
		double[] attVals = new double[numberDimensions + 1];
		double[] sample = null;
		Subspace s = null;
		// Model the change
		if (changeCounter >= 0) {
			if (changeCounter < changeLength) {
				if (useNewSubspaces()) {
					s = newSubspace;
					sample = newNormalDistribution.sample();
				} else {
					s = currentSubspace;
					sample = currentNormalDistribution.sample();
				}
				changeCounter++;
			} else {
				changeCounter = -1;
				currentNormalDistribution = newNormalDistribution;
				sample = currentNormalDistribution.sample();
				currentSubspace = newSubspace;
				s = currentSubspace;
			}
		} else {
			s = currentSubspace;
			sample = currentNormalDistribution.sample();
		}

		for (int i = 0; i < numberDimensions; i++) {
			if (s.contains(i)) {
				attVals[i] = centroid.centre[i] + sample[i];
			} else {
				attVals[i] = (instanceRandom.nextDouble() - 1) * scaleIrrelevant;
				// attVals[i] = -1;
			}
		}

		Instance inst = new DenseInstance(1.0, attVals);
		inst.setDataset(getHeader());
		inst.setClassValue(centroid.classLabel);

		return inst;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// super.prepareForUseImpl(monitor, repository);
		monitor.setCurrentActivity("Preparing subspace random RBF...", -1.0);
		this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
		this.modelRandom = new Random(this.modelRandomSeedOption.getValue());
		this.scaleIrrelevant = scaleIrrelevantDimensionsOption.getValue();
		numberDimensions = numAttsOption.getValue();
		numberSubspaces = numSubspacesOption.getValue();
		disjointSubspaces = disjointSubspacesOption.isSet();
		if (disjointSubspaces && sameSubspaceOption.isSet()
				&& numberSubspaces * avgSubspaceSizeOption.getValue() > numberDimensions) {
			throw new IllegalArgumentException("Too many subspace or dimensions per subspace.");
		}
		generateHeader();
		generateCentroids();
		setClasses();

		// Generate distribution
		this.sameSubspaceOption.set();
		double[] means = new double[numberDimensions];

		// Centroid weights
		numberCentroids = numCentroidsOption.getValue();

		SubspaceSet subspaceSet = createSubspaces();
		currentSubspace = createOneSubspace(subspaceSet);
		currentNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(subspaceSet, covariance));
		newNormalDistribution = null;
	}

	@Override
	public void restart() {
		generateCentroids();
		setClasses();
		this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
		this.modelRandom = new Random(this.modelRandomSeedOption.getValue());
		// Generate distribution
		double[] means = new double[numberDimensions];
		SubspaceSet subspaceSet = createSubspaces();
		currentSubspace = createOneSubspace(subspaceSet);
		currentNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(subspaceSet, covariance));
		newNormalDistribution = null;
	}

	private double[][] generateRandomCovarianceMatrix(SubspaceSet subspaceSet, double covariance) {
		double[][] covarianceMatrix = new double[numberDimensions][numberDimensions];
		for (int i = 0; i < numberDimensions; i++) {
			covarianceMatrix[i][i] = 1;
		}
		for (Subspace subspace : subspaceSet.getSubspaces()) {
			int[] dimensions = subspace.getDimensions();
			int d1;
			int d2;
			for (int i = 0; i < dimensions.length; i++) {
				for (int j = i + 1; j < dimensions.length; j++) {
					d1 = dimensions[i];
					d2 = dimensions[j];
					covarianceMatrix[d1][d2] = covariance;
					covarianceMatrix[d2][d1] = covariance;
				}
			}
		}

		return covarianceMatrix;
	}

	@Override
	public void subspaceChange(int changeLength) {
		SubspaceSet subspaceSet = createSubspaces();
		newSubspace = createOneSubspace(subspaceSet);
		double[] means = new double[numberDimensions];
		newNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(subspaceSet, covariance));
		changeCounter = 0;
		this.changeLength = changeLength;
	}

	private Subspace createOneSubspace(SubspaceSet subspaceSet) {
		Subspace s = new Subspace();
		for (Subspace subspace : subspaceSet.getSubspaces()) {
			for (Integer dim : subspace.getDimensions()) {
				s.addDimension(dim);
			}
		}
		return s;
	}

	private SubspaceSet createSubspaces() {
		SubspaceSet subspaceSet = new SubspaceSet();
		ArrayList<Integer> alreadyAdded = new ArrayList<Integer>();
		for (int i = 0; i < numberSubspaces; i++) {
			int numRelevantDims = 0;
			if (randomSubspaceSizeOption.isSet()) {
				numRelevantDims = (int) (avgSubspaceSizeOption.getValue() + (modelRandom.nextBoolean() ? -1 : 1)
						* avgSubspaceSizeOption.getValue() * modelRandom.nextDouble());
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
				int relevantDim = (int) (modelRandom.nextDouble() * (numAttsOption.getValue()));
				while (s.contains(relevantDim) || (disjointSubspaces && alreadyAdded.contains(relevantDim))) {
					relevantDim = (int) (modelRandom.nextDouble() * (numAttsOption.getValue()));
				}
				s.addDimension(relevantDim);
				alreadyAdded.add(relevantDim);
			}
			subspaceSet.addSubspace(s);
		}

		System.out.println("Correlated Subspaces: " + subspaceSet.toString());

		return subspaceSet;
	}
}

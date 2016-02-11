package streams;

import java.util.Random;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

public class DriftingGaussians extends SubspaceRandomRBFGeneratorDrift {

	/**
	 * 
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

	private final double covariance = 0.9;

	private Subspace currentSubspace;

	private Subspace newSubspace;

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
		//int centroidIndex = MiscUtils.chooseRandomIndexBasedOnWeights(this.centroidWeights, this.instanceRandom);
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
				attVals[i] = (instanceRandom.nextDouble() - 1) * 2;
				//attVals[i] = -1;
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
		generateHeader();
		generateCentroids();
		this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
		this.modelRandom = new Random(this.modelRandomSeedOption.getValue());
		numberDimensions = numAttsOption.getValue();
		// Generate distribution
		this.sameSubspaceOption.set();
		double[] means = new double[numberDimensions];
		
		//Centroid weights
		numberCentroids = numCentroidsOption.getValue();
		
		Subspace s = createSubspace();
		currentSubspace = s;
		currentNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(s, covariance));
		newNormalDistribution = null;
	}

	@Override
	public void restart() {
		generateCentroids();
		this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
		this.modelRandom = new Random(this.modelRandomSeedOption.getValue());
		// Generate distribution
		double[] means = new double[numberDimensions];
		Subspace s = createSubspace();
		currentSubspace = s;
		currentNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(s, covariance));
		newNormalDistribution = null;
	}

	private double[][] generateRandomCovarianceMatrix(Subspace s, double covariance) {
		double[][] covarianceMatrix = new double[numberDimensions][numberDimensions];
		for (int i = 0; i < numberDimensions; i++) {
			covarianceMatrix[i][i] = 1;
		}
		int[] dimensions = s.getDimensions();
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
		return covarianceMatrix;
	}

	@Override
	public void subspaceChange(int changeLength) {
		Subspace s = createSubspace();
		newSubspace = s;
		double[] means = new double[numberDimensions];
		newNormalDistribution = new MultivariateNormalDistribution(means,
				generateRandomCovarianceMatrix(s, covariance));
		changeCounter = 0;
		this.changeLength = changeLength;
	}
}

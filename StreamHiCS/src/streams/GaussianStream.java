package streams;

import java.util.ArrayList;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.streams.InstanceStream;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class GaussianStream implements InstanceStream {

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * The mean of the multivariate distribution.
	 */
	private double[] mean;
	/**
	 * The underlying stream generator.
	 */
	private MultivariateNormalDistribution normalDistribution;
	/**
	 * The header of the stream.
	 */
	private InstancesHeader streamHeader;
	private EuclideanDistance euclideanDistance;
	private double classRadius;

	public GaussianStream(double[] mean, double[][] covarianceMatrix, double classRadius) {
		if (mean == null) {
			// Mean will be initialised to be n x 0.0
			this.mean = new double[covarianceMatrix.length];
		} else {
			this.mean = mean;
		}
		init(covarianceMatrix);
		this.euclideanDistance = new EuclideanDistance();
		this.classRadius = classRadius;
	}

	public void setCovarianceMatrix(double[][] covarianceMatrix) {
		init(covarianceMatrix);
	}

	private void init(double[][] covarianceMatrix) {
		symmetryCheck(covarianceMatrix);
		normalDistribution = new MultivariateNormalDistribution(mean, covarianceMatrix);
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		int n = mean.length;
		for (int i = 0; i < n; i++) {
			attributes.add(new Attribute("normalAttribute" + i));
		}
		ArrayList<String> classLabels = new ArrayList<String>();
		classLabels.add("in");
		classLabels.add("out");
		attributes.add(new Attribute("class", classLabels));
		streamHeader = new InstancesHeader(new Instances("GaussianStream", attributes, 0));
		streamHeader.setClassIndex(n);
	}

	private void symmetryCheck(double[][] covarianceMatrix) {
		int m = covarianceMatrix.length;
		int n = covarianceMatrix[0].length;
		assert (m == n);

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				assert (covarianceMatrix[i][j] == covarianceMatrix[j][i]);
			}
		}
	}

	@Override
	public void getDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	public int measureByteSize() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public long estimatedRemainingInstances() {
		return 0;
	}

	@Override
	public InstancesHeader getHeader() {
		return this.streamHeader;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public boolean isRestartable() {
		return false;
	}

	@Override
	public Instance nextInstance() {
		double[] sample = normalDistribution.sample();
		InstancesHeader header = getHeader();
		Instance inst = new DenseInstance(header.numAttributes());
		for (int i = 0; i < mean.length; i++) {
			inst.setValue(i, sample[i]);
		}

		// Determine the class value
		double classValue;
		double distance = euclideanDistance.compute(sample, mean);
		if (distance <= classRadius) {
			classValue = 0;
		} else {
			classValue = 1;
		}
		inst.setDataset(header);
		inst.setClassValue(classValue);

		return inst;
	}

	@Override
	public void restart() {
		// TODO Auto-generated method stub
	}

	/**
	 * Returns the number of dimensions of this stream.
	 * 
	 * @return The number of dimensions of this stream.
	 */
	public int getNumberOfDimensions() {
		return mean.length;
	}

	@Override
	public MOAObject copy() {
		// TODO Auto-generated method stub
		return null;
	}
}

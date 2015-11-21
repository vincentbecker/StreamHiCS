package streams;

import java.util.ArrayList;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

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

	public GaussianStream(double[][] covarianceMatrix) {
		init(covarianceMatrix);
	}

	public void setCovarianceMatrix(double[][] covarianceMatrix) {
		init(covarianceMatrix);
	}

	private void init(double[][] covarianceMatrix) {
		symmetryCheck(covarianceMatrix);
		// Mean will be initialised to be n x 0.0
		mean = new double[covarianceMatrix.length];
		normalDistribution = new MultivariateNormalDistribution(mean, covarianceMatrix);
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < mean.length; i++) {
			attributes.add(new Attribute("normalAttribute" + i));
		}
		streamHeader = new InstancesHeader(new Instances("GaussianStream", attributes, 0));
	}

	private void symmetryCheck(double[][] covarianceMatrix) {
		int m = covarianceMatrix.length;
		int n = covarianceMatrix[0].length;
		assert(m == n);

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				assert(covarianceMatrix[i][j] == covarianceMatrix[j][i]);
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
		inst.setDataset(header);

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

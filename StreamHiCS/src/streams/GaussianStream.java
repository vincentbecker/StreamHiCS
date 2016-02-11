package streams;

import java.util.ArrayList;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.ml.distance.ManhattanDistance;

import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.streams.InstanceStream;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class represents a stream generator where the {@link Instance} are drawn
 * from a multinomial gaussian distribution. The class label is determined by
 * calculating the euclidean distance of the created {@link Instance} to the
 * mean of the distribution and comparing it to a fixed radius.
 * 
 * @author Vincent
 *
 */
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
	
	private double[][] covarianceMatrix;
	
	/**
	 * The {@link EuclideanDistance} instance. 
	 */
	private EuclideanDistance euclideanDistance;
	
	private ManhattanDistance manhattanDistance;
	
	/**
	 * The radius  to determine the label. 
	 */
	private double classRadius;

	/**
	 * Creates an instance of this class. 
	 * 
	 * @param mean The mean of the multinomial gaussian distribution
	 * @param covarianceMatrix The covariance matrix of the distribution
	 * @param classRadius The class radius
	 */
	public GaussianStream(double[] mean, double[][] covarianceMatrix, double classRadius) {
		if (mean == null) {
			// Mean will be initialised to be n x 0.0
			this.mean = new double[covarianceMatrix.length];
		} else {
			this.mean = mean;
		}
		init(covarianceMatrix);
		this.covarianceMatrix = covarianceMatrix;
		this.euclideanDistance = new EuclideanDistance();
		this.manhattanDistance = new ManhattanDistance();
		this.classRadius = classRadius;
	}

	/**
	 * Sets the covariance matrix. 
	 * 
	 * @param covarianceMatrix The new covariance matrix
	 */
	public void setCovarianceMatrix(double[][] covarianceMatrix) {
		this.covarianceMatrix = covarianceMatrix;
		init(covarianceMatrix);
	}

	/**
	 * Initialises the distribution. 
	 * 
	 * @param covarianceMatrix The covariance matrix
	 */
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

	/**
	 * Checks whether the covariance matrix is symmetric. 
	 * 
	 * @param covarianceMatrix THe covariance matrix
	 */
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
		//double distance = manhattanDistance.compute(sample, mean);
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
		normalDistribution = new MultivariateNormalDistribution(mean, covarianceMatrix);
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

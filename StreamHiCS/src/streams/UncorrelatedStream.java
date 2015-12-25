package streams;

import java.util.ArrayList;
import java.util.Random;

import moa.core.InstancesHeader;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.options.OptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class represents a stream generator where the values for each dimension
 * of an {@link Instance} is created randomly and independently from each other.
 * 
 * @author Vincent
 *
 */
public class UncorrelatedStream extends AbstractOptionHandler implements InstanceStream {

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -97884374978899603L;

	/**
	 * The number of dimensions.
	 */
	private int numberOfDimensions;

	/**
	 * The random generator.
	 */
	private Random random;

	/**
	 * The standard range for the random numbers generated is -0.5 to 0.5. This
	 * range can be scaled by a factor.
	 */
	private double scale;

	/**
	 * The header of the stream.
	 */
	private InstancesHeader streamHeader;

	/**
	 * The option determining the seed of the random generator.
	 */
	public IntOption randomSeedOption = new IntOption("randomSeed", 'r', "Seed for random generator.", 1, 1,
			Integer.MAX_VALUE);

	/**
	 * The option determining the number of dimensions of the stream.
	 */
	public IntOption dimensionsOption = new IntOption("numberOfDimensions", 'd', "Number of Dimensions.", 1, 1,
			Integer.MAX_VALUE);

	/**
	 * The option determining the scale of the range of the stream.
	 */
	public FloatOption scaleOption = new FloatOption("scale", 's', "Scale.", 1, 0, Double.MAX_VALUE);

	@Override
	public OptionHandler copy() {
		// TODO Auto-generated method stub
		return null;
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
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public InstancesHeader getHeader() {
		return streamHeader;
	}

	@Override
	public boolean hasMoreInstances() {
		return true;
	}

	@Override
	public boolean isRestartable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Instance nextInstance() {
		InstancesHeader header = getHeader();
		Instance inst = new DenseInstance(header.numAttributes());
		for (int i = 0; i < numberOfDimensions; i++) {
			inst.setValue(i, (random.nextDouble() - 0.5) * scale);
		}
		inst.setDataset(header);
		inst.setClassValue("NO_LABEL");

		return inst;
	}

	@Override
	public void restart() {
		prepareForUse();
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		this.random = new Random(this.randomSeedOption.getValue());
		this.numberOfDimensions = this.dimensionsOption.getValue();
		this.scale = this.scaleOption.getValue();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < numberOfDimensions; i++) {
			attributes.add(new Attribute("attribute" + i));
		}
		ArrayList<String> classLabels = new ArrayList<String>();
		classLabels.add("NO_LABEL");
		attributes.add(new Attribute("class", classLabels));
		streamHeader = new InstancesHeader(new Instances("UncorrelatedStream", attributes, 0));
		streamHeader.setClassIndex(numberOfDimensions);
	}

}

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

public class UncorrelatedStream extends AbstractOptionHandler implements InstanceStream{

	/**
	 * 
	 */
	private static final long serialVersionUID = -97884374978899603L;
	
	private int numberOfDimensions;
	private Random random;
	private double scale;

	/**
	 * The header of the stream.
	 */
	private InstancesHeader streamHeader;
	
	public IntOption randomSeedOption = new IntOption("randomSeed", 'r', "Seed for random generator.", 1, 1, Integer.MAX_VALUE);
	
	public IntOption dimensionsOption = new IntOption("numberOfDimensions", 'd', "Number of Dimensions.", 1, 1, Integer.MAX_VALUE);
	
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
			inst.setValue(i, (random.nextDouble() - 0.5)*scale);
		}
		inst.setDataset(header);

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
		streamHeader = new InstancesHeader(new Instances("UncorrelatedStream", attributes, 0));
	}

}

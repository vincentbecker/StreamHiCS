package streams;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomRBFGenerator;

/**
 * This class is a stream generator built from a {@link RandomRBFGenerator} and
 * adds one attribute which is exactly the same as the first attribute from the
 * {@link RandomRBFGenerator}'s {@link Instance}.
 * 
 * @author Vincent
 *
 */
public class CorrelatedStream implements InstanceStream {

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * The underlying {@link RandomRBFGenerator} stream generator.
	 */
	private RandomRBFGenerator rbfGen;
	
	/**
	 * The number of dimensions stored in the underlying stream generator.
	 */
	private int numberRBFDimensions;
	
	/**
	 * The header of the stream.
	 */
	private InstancesHeader streamHeader;

	/**
	 * Creates an instance of this class.
	 */
	public CorrelatedStream() {
		rbfGen = new RandomRBFGenerator();
		rbfGen.prepareForUse();
		numberRBFDimensions = rbfGen.getHeader().numAttributes();
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < numberRBFDimensions; i++) {
			attributes.add(new Attribute("rbfAttribute" + i));
		}
		// Add the 'new' attribute
		attributes.add(new Attribute("correlAttribute"));
		streamHeader = new InstancesHeader(new Instances("CorrelatedStream", attributes, 0));
	}

	@Override
	public MOAObject copy() {
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
		return rbfGen.estimatedRemainingInstances();
	}

	@Override
	public InstancesHeader getHeader() {
		return this.streamHeader;
	}

	@Override
	public boolean hasMoreInstances() {
		return rbfGen.hasMoreInstances();
	}

	@Override
	public boolean isRestartable() {
		return rbfGen.isRestartable();
	}

	@Override
	public Instance nextInstance() {
		// Copying an rbfInstance and appending one dimension which is fully
		// correlated to the first rbf dimension.
		Instance rbfInst = rbfGen.nextInstance();
		InstancesHeader header = getHeader();
		Instance inst = new DenseInstance(header.numAttributes());
		for (int i = 0; i < numberRBFDimensions; i++) {
			inst.setValue(i, rbfInst.value(i));
		}
		inst.setValue(numberRBFDimensions, rbfInst.value(0));
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
		return numberRBFDimensions + 1;
	}
}

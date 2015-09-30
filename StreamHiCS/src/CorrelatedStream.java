import weka.core.Instance;
import moa.MOAObject;
import moa.core.InstancesHeader;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomRBFGenerator;

public class CorrelatedStream implements InstanceStream {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private RandomRBFGenerator rbfGen;
	
	public CorrelatedStream(){
		rbfGen = new RandomRBFGenerator();
		rbfGen.prepareForUse();
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
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean hasMoreInstances() {
		return rbfGen.hasMoreInstances();
	}

	@Override
	public boolean isRestartable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public Instance nextInstance() {
		Instance rbfInst = rbfGen.nextInstance();
		
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void restart() {
		// TODO Auto-generated method stub

	}

}

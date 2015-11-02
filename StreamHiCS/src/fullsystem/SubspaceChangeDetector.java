package fullsystem;
import moa.classifiers.drift.SingleClassifierDrift;
import subspace.Subspace;
import weka.core.DenseInstance;
import weka.core.Instance;

public class SubspaceChangeDetector extends SingleClassifierDrift {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2008369880916696270L;

	private Subspace subspace;

	public SubspaceChangeDetector(Subspace subspace) {
		this.subspace = subspace;
	}
	
	public boolean isWarningDetected() {
		return (this.ddmLevel == DDM_WARNING_LEVEL);
	}
	 
	public boolean isChangeDetected() {
	    return (this.ddmLevel == DDM_OUTCONTROL_LEVEL);
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// Create new instance that only contains the dimensions of the subspace
		double[] subspaceData = new double[subspace.size() + 1];
		int l = inst.numAttributes() - 1;
		for (int i = 0; i < l; i++) {
			subspaceData[i] = inst.value(i);
		}
		// Getting the class label
		subspaceData[l] = inst.value(l);
		Instance subspaceInstance = new DenseInstance(inst.weight(), subspaceData);
		super.trainOnInstanceImpl(subspaceInstance);
	}
}

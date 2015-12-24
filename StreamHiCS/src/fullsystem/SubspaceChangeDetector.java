package fullsystem;

import java.util.ArrayList;

import moa.classifiers.drift.SingleClassifierDrift;
import moa.core.InstancesHeader;
import subspace.Subspace;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * This class represents a change detector using the DDM method which only takes
 * a {@link Subspace} into account.
 * 
 * @author Vincent
 *
 */
public class SubspaceChangeDetector extends SingleClassifierDrift implements ChangeDetector {

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -2008369880916696270L;

	/**
	 * The subsapce the {@link SubspaceChangeDetector} runs on.
	 */
	private Subspace subspace;

	/**
	 * The {@link InstancesHeader}.
	 */
	private InstancesHeader header;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param subspace
	 *            The {@link Subspace} the {@link SubspaceChangeDetector} should
	 *            run on.
	 */
	public SubspaceChangeDetector(Subspace subspace) {
		this.subspace = subspace;
	}

	/**
	 * Returns the {@link Subspace} this {@link SubspaceChangeDetector} runs on.
	 * 
	 * @return The {@link Subspace} this {@link SubspaceChangeDetector} runs on.
	 */
	public Subspace getSubspace() {
		return this.subspace;
	}

	public boolean isWarningDetected() {
		return (this.ddmLevel == DDM_WARNING_LEVEL);
	}

	public boolean isChangeDetected() {
		return (this.ddmLevel == DDM_OUTCONTROL_LEVEL);
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		double classVal = inst.classValue();
		// Create new instance that only contains the dimensions of the subspace
		int l = subspace.size();
		if (header == null) {
			// set the header
			// Copy all the appropriate attributes
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			for (int i = 0; i < l; i++) {
				attributes.add(inst.dataset().attribute(subspace.getDimension(i)));
			}
			// set the class attribute
			attributes.add(inst.dataset().attribute(inst.classIndex()));
			header = new InstancesHeader(new Instances("subspaceData", attributes, 0));
			header.setClassIndex(l);
		}
		double[] subspaceData = new double[l + 1];
		for (int i = 0; i < l; i++) {
			subspaceData[i] = inst.value(subspace.getDimension(i));
		}
		// Setting the class label
		subspaceData[l] = inst.value(inst.classIndex());
		Instance subspaceInstance = new DenseInstance(inst.weight(), subspaceData);
		subspaceInstance.setDataset(header);
		double classVal2 = subspaceInstance.classValue();
		if (classVal != classVal2) {
			System.out.println("Class values different: " + classVal + " != " + classVal2);
		}
		super.trainOnInstanceImpl(subspaceInstance);
	}
}

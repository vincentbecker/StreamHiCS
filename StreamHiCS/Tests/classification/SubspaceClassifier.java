package classification;

import java.util.ArrayList;

import fullsystem.SubspaceChangeDetector;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstancesHeader;
import subspace.Subspace;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class SubspaceClassifier extends HoeffdingTree {
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
	public SubspaceClassifier(Subspace subspace) {
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

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		//double classVal = inst.classValue();
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
		/*
		double classVal2 = subspaceInstance.classValue();
		if (classVal != classVal2) {
			System.out.println("Class values different: " + classVal + " != " + classVal2);
		}
		*/
		super.trainOnInstanceImpl(subspaceInstance);
	}
}

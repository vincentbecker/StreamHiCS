package classification;

import java.util.ArrayList;
import java.util.BitSet;

import fullsystem.Callback;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import fullsystem.SubspaceChangeDetector;
import fullsystem.SubspaceClassifier;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import subspace.Subspace;
import subspace.SubspaceSet;
import weka.core.Instance;
import weka.core.Utils;

public class CorrelatedSubspacesClassifier extends AbstractClassifier implements Callback {

	/**
	 * The serial version ID.
	 */
	private static final long serialVersionUID = -686978424502737299L;

	/**
	 * The option determining the value of initial point before the correlated
	 * {@link Subspace}s are evaluated the first time.
	 */
	public IntOption initOption = new IntOption("init", 'i',
			"The number of instances after which the correlated subspaces are evaluated the first time.", 500, 1,
			Integer.MAX_VALUE);

	/**
	 * The {@link StreamHiCS} instance.
	 */
	private StreamHiCS streamHiCS;

	/**
	 * The {@link FullSpaceChangeDetector} instance.
	 */
	private HoeffdingTree fullSpaceClassifier;

	/**
	 * The {@link ArrayList} containing the {@link SubspaceChangeDetector}s,
	 * i.e. the change detector running on each correlated {@link Subspace}.
	 */
	private ArrayList<SubspaceClassifier> subspaceClassifiers;

	private RestSpaceClassifier restSpaceClassifier;

	/**
	 * The number of dimensions.
	 */
	private int numberOfDimensions;

	/**
	 * The number of samples seen.
	 */
	private int numberSamples;

	/**
	 * A flag showing if the an evaluation of the correlated subspace has been
	 * carried out already.
	 */
	private boolean initialized = false;

	/**
	 * The number of instances received before evaluating the correlated
	 * subspaces the first time.
	 */
	private int numberInit;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param numberOfDimensions
	 *            The number of dimensions.
	 * @param streamHiCS
	 *            The {@link StreamHiCS} instance.
	 */
	public CorrelatedSubspacesClassifier(int numberOfDimensions, StreamHiCS streamHiCS) {
		this.numberOfDimensions = numberOfDimensions;
		this.streamHiCS = streamHiCS;
	}

	@Override
	public void onAlarm() {
		// Change subspaces and reset ChangeDetection
		// Happens anyway: fullSpaceChangeDetector.resetLearning();
		SubspaceSet correlatedSubspaces =  streamHiCS.getCurrentlyCorrelatedSubspaces();
		/*
		SubspaceSet correlatedSubspaces = new SubspaceSet();
		if (numberSamples < 18000) {
			Subspace s1 = new Subspace(3, 4, 7);
			Subspace s2 = new Subspace(5, 6, 7);
			correlatedSubspaces.addSubspace(s1);
			correlatedSubspaces.addSubspace(s2);
		} else {
			Subspace s1 = new Subspace(3, 4, 5, 6, 7);
			correlatedSubspaces.addSubspace(s1);
		}
		*/
		// System.out.println("Number of samples: " + numberSamples);
		System.out.println("Correlated: " + correlatedSubspaces.toString() + " at " + numberSamples);
		// subspaceChangeDetectors.clear();
		/*
		 * System.out.println("Number of samples: " + numberSamples);
		 * System.out.println("Number of microclusters: " +
		 * streamHiCS.getNumberOfElements()); System.out.println("Correlated: "
		 * + streamHiCS.toString()); for (Subspace s :
		 * correlatedSubspaces.getSubspaces()) {
		 * System.out.print(s.getContrast() + ", "); } System.out.println();
		 */
		ArrayList<SubspaceClassifier> temp = new ArrayList<SubspaceClassifier>();
		// Mark which dimensions are contained in subspaces to add detectors for
		// all the single dimensions which are not contained in subspaces.
		BitSet marked = new BitSet(numberOfDimensions);
		boolean found = false;
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			for (int d : s.getDimensions()) {
				marked.set(d);
			}
			found = false;
			for (SubspaceClassifier sc : subspaceClassifiers) {
				// If there is an 'old' change detector running on the subspace
				// already we continue using that
				if (s.equals(sc.getSubspace())) {
					temp.add(sc);
					found = true;
				}
			}
			// If the subspace is new we start a new change detector on it
			if (!found) {
				SubspaceClassifier sc = createSubspaceClassifier(s);
				temp.add(sc);
			}
		}
		// Handle the one-dimensional subspaces
		Subspace restSpace = new Subspace();
		for (int i = 0; i < numberOfDimensions; i++) {
			/*
			 * if (!marked.get(i)) { found = false; Subspace s = new
			 * Subspace(i); for (SubspaceChangeDetector scd :
			 * subspaceChangeDetectors) { // If there is an 'old' change
			 * detector running on the // subspace // already we continue using
			 * that if (scd.getSubspace().size() == 1 &&
			 * s.equals(scd.getSubspace())) { temp.add(scd); found = true; } }
			 * if (!found) { SubspaceChangeDetector scd =
			 * createSubspaceChangeDetector(s); temp.add(scd); } }
			 */

			if (!marked.get(i)) {
				restSpace.addDimension(i);
			}

		}
		if (!restSpace.isEmpty()) {
			restSpaceClassifier = new RestSpaceClassifier(restSpace);
			restSpaceClassifier.prepareForUse();
		}
		subspaceClassifiers = temp;
		initialized = true;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		streamHiCS.add(inst);
		if (inst.classValue() < 0) {
			System.out.println("Class value < 0");
		}
		if (!initialized) {
			fullSpaceClassifier.trainOnInstance(inst);
		}
		numberSamples++;
		if (numberSamples == numberInit) {
			// Evaluate the correlated subspaces once
			init();
		}
		for (SubspaceClassifier sc : subspaceClassifiers) {
			sc.trainOnInstance(inst);
		}
		if (restSpaceClassifier != null) {
			restSpaceClassifier.trainOnInstance(inst);
		}
	}

	/**
	 * Calculate the correlated {@link Subspace}s and adds a change detector for
	 * each of them, if there are any
	 */
	private void init() {
		SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
		BitSet marked = new BitSet(numberOfDimensions);
		for (Subspace s : correlatedSubspaces.getSubspaces()) {
			for (int d : s.getDimensions()) {
				marked.set(d);
			}
			SubspaceClassifier sc = createSubspaceClassifier(s);
			subspaceClassifiers.add(sc);
		}
		Subspace restSpace = new Subspace();
		for (int i = 0; i < numberOfDimensions; i++) {
			if (!marked.get(i)) {
				restSpace.addDimension(i);
				/*
				 * Subspace s = new Subspace(); s.addDimension(i);
				 * SubspaceChangeDetector scd = createSubspaceChangeDetector(s);
				 * subspaceChangeDetectors.add(scd);
				 */
			}
		}
		if (!restSpace.isEmpty()) {
			restSpaceClassifier = new RestSpaceClassifier(restSpace);
			restSpaceClassifier.prepareForUse();
		}

		initialized = true;
	}

	/**
	 * Creates and initialises a {@link SubspaceChangeDetector} for a given
	 * {@link Subspace}.
	 * 
	 * @param s
	 *            The subspace
	 * @return The {@link SubspaceChangeDetector}.
	 */
	private SubspaceClassifier createSubspaceClassifier(Subspace s) {
		SubspaceClassifier sc = new SubspaceClassifier(s);
		sc.prepareForUse();

		return sc;
	}

	@Override
	public void prepareForUseImpl(TaskMonitor arg0, ObjectRepository arg1) {
		// ChangeDetectors
		this.fullSpaceClassifier = new HoeffdingTree();
		fullSpaceClassifier.prepareForUse();
		// No correlated subspaces yet.
		this.subspaceClassifiers = new ArrayList<SubspaceClassifier>();
		this.numberInit = initOption.getValue();
		super.prepareForUseImpl(arg0, arg1);
	}

	/**
	 * Returns the number of elements held in {@link StreamHiCS}.
	 * 
	 * @return The number of elements held in {@link StreamHiCS}.
	 */
	public int getNumberOfElements() {
		return streamHiCS.getNumberOfElements();
	}

	/**
	 * Returns the currently correlated {@link Subspace}s in a
	 * {@link SubspaceSet}.
	 * 
	 * @return The currently correlated {@link Subspace}s in a
	 *         {@link SubspaceSet}.
	 */
	public SubspaceSet getCurrentlyCorrelatedSubspaces() {
		return streamHiCS.getCurrentlyCorrelatedSubspaces();
	}

	@Override
	public void resetLearningImpl() {
		fullSpaceClassifier.resetLearning();
		numberSamples = 0;
		subspaceClassifiers.clear();
		streamHiCS.clear();
		initialized = false;
	}

	@Override
	public double[] getVotesForInstance(Instance arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder arg0, int arg1) {
		// TODO Auto-generated method stub

	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getClassPrediction(Instance instance) {
		if (numberSamples <= numberInit) {
			return Utils.maxIndex(fullSpaceClassifier.getVotesForInstance(instance));
		} else {
			int numberClasses = instance.classAttribute().numValues();
			double[] totalPredictions = new double[numberClasses];
			
			for (SubspaceClassifier sc : subspaceClassifiers) {
				double[] scdPrediction = sc.getVotesForInstance(instance);
				if (scdPrediction.length == numberClasses) {
					for (int i = 0; i < numberClasses; i++) {
						totalPredictions[i] += scdPrediction[i];
					}
				}
			}
			if (restSpaceClassifier != null) {
				double[] prediction = restSpaceClassifier.getVotesForInstance(instance);
				if (prediction.length == numberClasses) {
					for (int i = 0; i < numberClasses; i++) {
						totalPredictions[i] += prediction[i];
					}
				}
			}
			return Utils.maxIndex(totalPredictions);
		}
	}
}

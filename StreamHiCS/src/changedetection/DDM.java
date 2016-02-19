package changedetection;

import moa.classifiers.core.driftdetection.AbstractChangeDetector;
import moa.core.ObjectRepository;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;

public class DDM extends AbstractChangeDetector {

	private static final long serialVersionUID = -3518369648142099719L;

	// private static final int DDM_MINNUMINST = 30;
	public IntOption minNumInstancesOption = new IntOption("minNumInstances", 'n',
			"The minimum number of instances before permitting detecting change.", 30, 0, Integer.MAX_VALUE);
	private int m_n;

	private double m_p;

	private double m_s;

	private double m_psmin;

	private double m_pmin;

	private double m_smin;

	public DDM() {
		resetLearning();
	}

	@Override
	public void resetLearning() {
		m_n = 1;
		m_p = 1;
		m_s = 0;
		m_psmin = Double.MAX_VALUE;
		m_pmin = Double.MAX_VALUE;
		m_smin = Double.MAX_VALUE;
	}

	@Override
	public void input(double prediction) {
		// prediction must be 1 or 0
		// It monitors the error rate
		if (this.isChangeDetected == true || this.isInitialized == false) {
			resetLearning();
			this.isInitialized = true;
		}

		m_p = m_p + (prediction - m_p) / (double) m_n;
		m_s = Math.sqrt(m_p * (1 - m_p) / (double) m_n);
		m_n++;

		// System.out.print(prediction + " " + m_n + " " + (m_p+m_s) + " ");
		this.estimation = m_p;
		this.isChangeDetected = false;
		this.isWarningZone = false;
		this.delay = 0;

		if (m_n < this.minNumInstancesOption.getValue()) {
			return;
		}

		/*
		if(m_psmin == 0.0 && m_p + m_s > 0.0 ){
			m_pmin = m_p;
			m_smin = m_s;
			m_psmin = m_p + m_s;
		}
		*/
		
		if (m_p + m_s <= m_psmin) {
			m_pmin = m_p;
			m_smin = m_s;
			m_psmin = m_p + m_s;
		}

			if (m_n > this.minNumInstancesOption.getValue() && m_p + m_s > m_pmin + 3 * m_smin) {
				// System.out.println(m_p + ",D");
				this.isChangeDetected = true;
				if (m_pmin == 0.0) {
					//System.out.println("m_pmin = " + 0.0);
				}
				// resetLearning();
			} else if (m_p + m_s > m_pmin + 2 * m_smin) {
				// System.out.println(m_p + ",W");
				this.isWarningZone = true;
			} else {
				this.isWarningZone = false;
				// System.out.println(m_p + ",N");
			}
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
	}
}

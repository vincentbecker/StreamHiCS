package changechecker;

import fullsystem.Callback;

/**
 * This class represents a system that checks for changes and in case of a
 * change notifies the callback. The general procedure is to regularly check a
 * condition which controls for the change. This condition is implemented by the
 * subclasses.
 * 
 * @author Vincent
 *
 */
public abstract class ChangeChecker {

	/**
	 * The time counter
	 */
	private int time = 0;

	/**
	 * The interval the change condition should be checked.
	 */
	private int checkInterval;

	/**
	 * The {@link Callback} to inform in case of a change.
	 */
	protected Callback callback;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param checkInterval
	 *            The time interval after which to check for a change
	 */
	public ChangeChecker(int checkInterval) {
		this.checkInterval = checkInterval;
	}

	/**
	 * Sets the {@link Callback}.
	 * 
	 * @param callback
	 *            The callback
	 */
	public void setCallback(Callback callback) {
		this.callback = callback;
	}

	/**
	 * Checks the condition after the check interval has passed and informs the
	 * {@link Callback} in case of a change. The calling object should call this
	 * method every time an instance is added.
	 */
	public void poll() {
		time++;
		if (time % checkInterval == 0) {
			time = 0;
			if (checkForChange()) {
				callback.onAlarm();
			}
		}
	}

	/**
	 * The condition for a change implemented by the subclasses.
	 * 
	 * @return Returns true, if a change was detected, false otherwise.
	 */
	public abstract boolean checkForChange();
}

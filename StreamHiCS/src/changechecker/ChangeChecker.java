package changechecker;

import fullsystem.Callback;

public abstract class ChangeChecker {

	private int time = 0;
	private int checkInterval;
	protected Callback callback;

	public ChangeChecker(int checkInterval) {
		this.checkInterval = checkInterval;
	}

	public void setCallback(Callback callback) {
		this.callback = callback;
	}

	public void poll() {
		time++;
		if (time % checkInterval == 0) {
			time = 0;
			if(checkForChange()){
				callback.onAlarm();
			}
		}
	}

	public abstract boolean checkForChange();
}

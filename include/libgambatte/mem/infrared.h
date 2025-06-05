//
//   Copyright (C) 2007-2010 by sinamas <sinamas at users.sourceforge.net>
//
//   This program is free software; you can redistribute it and/or modify
//   it under the terms of the GNU General Public License version 2 as
//   published by the Free Software Foundation.
//
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License version 2 for more details.
//
//   You should have received a copy of the GNU General Public License
//   version 2 along with this program; if not, write to the
//   Free Software Foundation, Inc.,
//   51 Franklin St, Fifth Floor, Boston, MA  02110-1301, USA.
//

#ifndef INFRARED_H
#define INFRARED_H

#include "remote.h"
#include "savestate.h"
#include "newstate.h"

namespace gambatte {

class Infrared {
public:
	enum WhichIrGb { this_gb = 0, linked_gb = 1, remote = 2 };

	Infrared();
	bool getIrTrigger() const { return irTrigger_; };
	void ackIrTrigger() { irTrigger_ = false; };

	bool getIrSignal(WhichIrGb which, unsigned long const cc);
	void setIrSignal(WhichIrGb which, bool signal);

	void setRemoteCallback(unsigned char (*callback)()) { remote_.setRemoteCallback(callback); }
	void setRemoteActive(bool active) { remote_.setRemoteActive(active); }
	void resetCc(unsigned long const oldCc, unsigned long const newCc) { remote_.resetCc(oldCc, newCc); }
	void speedChange(unsigned long const cc) { remote_.speedChange(cc); }

	void saveState(SaveState &state) const;
	void loadState(SaveState const &state, bool const ds);
	template<bool isReader>void SyncState(NewState *ns);

private:
	bool irTrigger_;
	bool thisGbIrSignal_;
	bool linkedGbIrSignal_;
	Remote remote_;
};

}

#endif

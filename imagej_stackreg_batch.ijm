// Arguments, separated by "|":
// ChanA input | ChanB input | ChanA output | ChanB output | transform file
args = split(getArgument(), "|");
if (args.length != 5)
    exit("Expected five pipe-separated arguments");

setBatchMode(true);

open(args[1]);
rename("STACKREG_CHAN_B");
setSlice(1);

open(args[0]);
rename("STACKREG_CHAN_A");
setSlice(1);

run("MultiStackReg",
    "stack_1=[STACKREG_CHAN_B] action_1=Align file_1=[" + args[4] + "] " +
    "stack_2=[STACKREG_CHAN_A] action_2=[Load Transformation File] file_2=[] " +
    "transformation=Translation save");

selectWindow("STACKREG_CHAN_A");
saveAs("Tiff", args[2]);
close();

selectWindow("STACKREG_CHAN_B");
saveAs("Tiff", args[3]);
close();

setBatchMode(false);

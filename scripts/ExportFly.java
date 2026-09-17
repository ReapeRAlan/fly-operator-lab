// Export decompiled game routines for the local interoperability investigation.
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
public class ExportFly extends GhidraScript {
  public void run() throws Exception {
    String[] args = getScriptArgs();
    DecompInterface decomp = new DecompInterface();
    decomp.openProgram(currentProgram);
    for (String row : Files.readAllLines(Paths.get(args[0]), StandardCharsets.UTF_8)) {
      if (row.isBlank()) continue;
      String[] parts = row.split("\t");
      Address addr = toAddr(Long.parseUnsignedLong(parts[0],16));
      Function fn = getFunctionAt(addr);
      if (fn == null) { disassemble(addr); fn = createFunction(addr,parts[1].replace("::","_")); }
      if (fn == null) { println("NO_FUNCTION " + row); continue; }
      DecompileResults result = decomp.decompileFunction(fn,120,monitor);
      if (result.decompileCompleted()) {
        Path dest=Paths.get(args[1], parts[1].replace("::","_")+".c");
        Files.writeString(dest,result.getDecompiledFunction().getC(),StandardCharsets.UTF_8);
        println("EXPORTED " + dest);
      } else println("FAILED " + row + " " + result.getErrorMessage());
    }
    decomp.dispose();
  }
}

/* ARM MPS2-AN500 (Cortex-M7) Memory Map for QEMU bare-metal execution proof.
 *
 * FLASH -> ZBT SSRAM1 (0x0000_0000, 4096K): vector table, .text, .rodata
 * RAM   -> ZBT SSRAM2/3 (0x2000_0000, 4096K): .data, .bss, stack
 */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4096K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4096K
}

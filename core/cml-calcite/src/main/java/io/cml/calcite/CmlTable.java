/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.cml.calcite;

import org.apache.calcite.DataContext;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.ScannableTable;
import org.apache.calcite.schema.impl.AbstractTable;

public class CmlTable
        extends AbstractTable
        implements ScannableTable
{
    private final String name;
    private final RelDataType rowType;

    public CmlTable(String name, RelDataType rowType)
    {
        this.rowType = rowType;
        this.name = name;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory)
    {
        return rowType;
    }

    @Override
    public Enumerable<Object[]> scan(DataContext root)
    {
        // we don't need to scan table data
        return null;
    }

    public String getName()
    {
        return name;
    }
}